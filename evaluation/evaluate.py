#!/usr/bin/env python3

from openai import OpenAI
from textwrap import dedent

import argparse
import yaml
import json
import sys

sys.path.append(".")
from data.dataset import OpenWeather, IceHockey, GSMArena, Wikidata, OurWorldInData
from pathlib import Path
import os
import coloredlogs
import logging
import time
import requests

# logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Metric:
    def __init__(self, metric_name, load_args=None):
        self.metric_name = metric_name

    def load_dataset(self, dataset_name, base_path):
        dataset_dict = {
            "openweather": OpenWeather,
            "ice_hockey": IceHockey,
            "gsmarena": GSMArena,
            "wikidata": Wikidata,
            "owid": OurWorldInData,
        }
        return dataset_dict[dataset_name](base_path=base_path)

    def create_annotation(self, text, j, table_idx):
        annotation_list = []
        current_pos = 0

        for error in j["errors"]:
            # find the `start` index of the error in the text
            start_pos = text.lower().find(error["text"].lower(), current_pos)

            if current_pos != 0 and start_pos == -1:
                # try from the beginning
                start_pos = text.find(error["text"])

            if start_pos == -1:
                logger.warning(f"Cannot find error {error} in text {text}, skipping")
                continue

            error["start"] = start_pos
            annotation_list.append(error)

            current_pos = start_pos + len(error["text"])

        annotation = {
            "annotator_id": self.metric_name,
            "dataset": self.dataset_name,
            "model": self.model_name,
            "setup": self.setup_name,
            "split": self.split,
            "table_idx": table_idx,
            "annotations": annotation_list,
        }

        return annotation

    def run(self, model_name, dataset_name, setup_name, split, base_path, debug):
        self.debug = debug
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset = self.load_dataset(dataset_name, base_path)
        self.setup_name = setup_name
        self.setup = setup_name
        self.split = split

        data = self.dataset.get_data(split=split)
        annotations = []

        if self.debug:
            data = data[:3]

        for i, data_input in enumerate(data):
            text_outputs = self.dataset.get_generated_outputs(split, i)

            text_outputs = [
                x
                for x in text_outputs
                if x["setup"]["name"] == self.setup_name
                and x["model"] == self.model_name
            ]
            for out in text_outputs:
                text = out["generated"]

                try:
                    logger.info(
                        f"{self.dataset_name} | {self.setup_name} | {self.model_name} | {i+1}/{len(data)}"
                    )
                    j = self.annotate_example(data=data_input, text=text)
                    annotation = self.create_annotation(text, j, table_idx=i)
                    annotations.append(annotation)

                    logger.info("=" * 80)
                except Exception as e:
                    logger.error(f"Error while annotating example: {e}")
                    logger.error(f"Example: {text}")
                    logger.error(f"Data: {data_input}")

        out_dir = f"{base_path}/annotations/gpt-4"
        start = int(time.time())
        filename = (
            f"gpt-4-{dataset_name}-{split}-{model_name}-{setup_name}-{start}.jsonl"
        )

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(out_dir, filename), "w") as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + "\n")


class GPT4Metric(Metric):
    def __init__(self, load_args=None):
        # super().__init__("gpt-3.5-turbo-1106", load_args)
        super().__init__("gpt-4-1106-preview", load_args)
        self.client = OpenAI()

        with open("evaluation/gpt4_metric.yaml") as f:
            config = yaml.safe_load(f)

        self.system_msg = config["system_msg"]
        self.metric_prompt_template = config["prompt_template"]

    def annotate_example(self, data, text):
        try:
            prompt = self.metric_prompt_template.format(data=data, text=text)

            response = self.client.chat.completions.create(
                model=self.metric_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            annotation_str = response.choices[0].message.content

            j = json.loads(annotation_str)
            logger.info(j)
            return j
        except Exception as e:
            logger.error(e)
            return {"errors": []}


class Llama3Metric(Metric):
    def __init__(self, load_args=None):
        super().__init__("gpt-4-llama3", load_args)

        with open("evaluation/gpt4_metric.yaml") as f:
            config = yaml.safe_load(f)

        self.system_msg = config["system_msg"]
        self.metric_prompt_template = config["prompt_template"]

    def annotate_example(self, data, text):
        try:
            prompt = self.metric_prompt_template.format(data=data, text=text)

            response = requests.post(
                "http://tdll-3gpu2.ufal.hide.ms.mff.cuni.cz:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"seed": 42, "temperature": 0},
                },
            )
            annotation_str = response.json()["response"].strip()
            j = json.loads(annotation_str)

            # the model often tends to produce a nested list
            if type(j["errors"][0]) == list:
                j["errors"] = j["errors"][0]

            logger.info(j)
            return j
        except Exception as e:
            logger.error(e)
            return {"errors": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["openweather", "ice_hockey", "gsmarena", "wikidata", "owid"],
        help="Dataset",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["mistral", "zephyr", "llama2"],
        default="mistral",
        help="Model",
    )
    parser.add_argument(
        "-p",
        "--base-path",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "-s",
        "--setup",
        type=str,
        help=f"Setup (parameters, prompt, etc.)",
    )
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev", help="Dataset split"
    )
    parser.add_argument("--debug", action="store_true", help="Debug run")
    args = parser.parse_args()

    e = GPT4Metric()

    e.run(
        model_name=args.model,
        dataset_name=args.dataset,
        setup_name=args.setup,
        split=args.split,
        base_path=args.base_path,
        debug=args.debug,
    )
