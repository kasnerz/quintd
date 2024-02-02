#!/usr/bin/env python3

import argparse
import yaml
import json
import os
import coloredlogs
import logging
import sys

sys.path.append(".")
from data.dataset import OpenWeather, IceHockey, GSMArena, Wikidata, OurWorldInData
from model.models import (
    MistralInstructModel,
    ZephyrBetaModel,
    Llama2Model,
    ChatGPTModel,
)
from pathlib import Path

# logging.basicConfig(format="%(message)s", level=logging.INFO, datefmt="%H:%M:%S")
coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Setup:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    SETUP_DIR = os.path.join(dir_path, "setups")

    def __init__(self, name, seed_strategy="static"):
        self.name = name
        self.default_params = {
            "auto_max_new_tokens": False,
            "mode": "instruct",
            "truncation_length": 32768,
        }
        self.seed_strategy = seed_strategy
        setup_filename = f"{self.__class__.SETUP_DIR}/{name}.yaml"

        if not Path(setup_filename).exists():
            raise ValueError(f"Error: parameters file {setup_filename} does not exist.")

        with open(setup_filename) as f:
            self.settings = yaml.load(f, Loader=yaml.FullLoader)

        self.settings["params"].update(self.default_params)
        self.settings["name"] = self.name

    def get_prompt(self, dataset, data_input):
        prompt_template = self.settings["prompt"]
        prompt_args = self.get_map(dataset.name)
        prompt_args["DATA"] = data_input

        prompt = prompt_template.format_map(prompt_args)

        return prompt

    def get_map(self, dataset_name):
        map_dict = {}

        for key in ["output", "style", "aspect"]:
            if key in self.settings:
                map_dict[key] = self.settings[key][dataset_name]

        return map_dict

    def get_settings(self):
        return self.settings

    def get_model_params(self):
        params = self.settings["params"].copy()
        return params

    def get_model_input(self, dataset, data_input, step):
        prompt = self.get_prompt(dataset, data_input)
        start_with = self.settings["start_with"].format_map(self.get_map(dataset.name))

        params = self.get_model_params()

        if self.seed_strategy == "dynamic":
            # change deterministically random seed for every example
            params["seed"] += step

        return prompt, params, start_with


class Processor:
    def __init__(self, model_name, api_url):
        self.model_name = model_name
        self.model = self.load_model(model_name, api_url)

    def load_model(self, model_name, api_url):
        model_dict = {
            "mistral": MistralInstructModel,
            "zephyr": ZephyrBetaModel,
            "llama2": Llama2Model,
            "gpt-3.5": ChatGPTModel,
        }
        return model_dict[model_name](api_url=api_url)

    def load_dataset(self, dataset_name, base_path):
        dataset_dict = {
            "openweather": OpenWeather,
            "ice_hockey": IceHockey,
            "gsmarena": GSMArena,
            "wikidata": Wikidata,
            "owid": OurWorldInData,
        }
        return dataset_dict[dataset_name](base_path=base_path)

    def run(self, dataset_name, setup_name, split, base_path, debug):
        self.debug = debug
        self.dataset_name = dataset_name
        self.dataset = self.load_dataset(dataset_name, base_path)
        self.setup = Setup(name=setup_name)
        self.setup_name = setup_name

        outputs = {}
        outputs["dataset"] = self.dataset_name
        outputs["model"] = self.model_name
        outputs["setup"] = self.setup.get_settings()
        outputs["generated"] = []

        data = self.dataset.get_data(split=split)
        out_dir = f"{base_path}/outputs/{split}/{self.dataset_name}/{self.setup_name}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if self.debug:
            data = data[:3]

        for i, data_input in enumerate(data):
            prompt, params, start_with = self.setup.get_model_input(
                self.dataset, data_input, i
            )

            # tokenizing solely for length calculation
            tokenized_prompt = self.model.tokenize(prompt)["input_ids"][0]
            in_token_cnt = len(tokenized_prompt)
            logger.info(
                f"{self.dataset_name} | {self.setup_name} | {self.model_name} | {i+1}/{len(data)} | {in_token_cnt} tok."
            )

            model_output = self.model.generate(prompt, params, start_with)
            out_token_cnt = len(self.model.tokenize(model_output)["input_ids"][0])

            logger.info(model_output)
            logger.info("=" * 80)
            outputs["generated"].append(
                {
                    "in": prompt,
                    "out": model_output,
                    "tokens": {"in": in_token_cnt, "out": out_token_cnt},
                }
            )

        base_filename = f"{out_dir}/{self.model_name}"

        # full output JSON
        with open(f"{base_filename}.json", "w") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

        # one output per line
        with open(f"{base_filename}.out", "w") as f:
            for output in outputs["generated"]:
                out = output["out"]
                out = out.replace("\r", "").replace("\n", "\\n")

                f.write(out + "\n")


if __name__ == "__main__":
    setup_choices = [x.stem for x in Path(Setup.SETUP_DIR).glob("*.yaml")]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["openweather", "ice_hockey", "gsmarena", "wikidata", "owid"],
        help="Dataset",
    )
    parser.add_argument(
        "-a",
        "--api-url",
        type=str,
        required=True,
        help="URL for the text-generation-webui API: <server>:<port>/api",
    )
    parser.add_argument(
        "-p",
        "--base-path",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["mistral", "zephyr", "llama2", "gpt-3.5"],
        default="mistral",
        help="Model",
    )
    parser.add_argument(
        "-s",
        "--setup",
        choices=setup_choices,
        help=f"Setup (parameters, prompt, etc.), loaded from {Setup.SETUP_DIR}/*.yaml",
    )
    parser.add_argument(
        "--split", choices=["dev", "test"], default="dev", help="Dataset split"
    )
    parser.add_argument("--debug", action="store_true", help="Debug run")
    args = parser.parse_args()

    p = Processor(model_name=args.model, api_url=args.api_url)

    p.run(
        dataset_name=args.dataset,
        setup_name=args.setup,
        split=args.split,
        base_path=args.base_path,
        debug=args.debug,
    )
