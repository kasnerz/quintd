#!/usr/bin/env python3

import re
import glob
import json
import random
import os
import pandas as pd
from collections import defaultdict

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import OpenWeather, IceHockey, GSMArena, Wikidata, OurWorldInData

dataset_dict = {
    "openweather": OpenWeather,
    "ice_hockey": IceHockey,
    "gsmarena": GSMArena,
    "wikidata": Wikidata,
    "owid": OurWorldInData,
}

ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__))


is_missing = 0
is_off_topic = 0

jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "human", f"*.jsonl"))
for jsonl_file in jsonl_files:
    new_lines = []
    with open(jsonl_file) as f:
        for line in f:
            annotation_record = json.loads(line)

            dataset = dataset_dict[annotation_record["dataset"]]()
            outputs = dataset.get_generated_outputs(
                annotation_record["split"], annotation_record["table_idx"]
            )

            new_annotations = []
            annotations = sorted(
                annotation_record["annotations"], key=lambda x: x["start"]
            )

            is_missing += int(annotation_record["flags"]["is_missing"])
            is_off_topic += int(annotation_record["flags"]["is_off_topic"])

            # merge the annotation if they have consecutive spans
            for annotation in annotations:
                if len(new_annotations) == 0:
                    new_annotations.append(annotation)
                else:
                    prev_annotation = new_annotations[-1]
                    prev_end = prev_annotation["start"] + len(prev_annotation["text"])
                    if (
                        prev_end + 1 >= annotation["start"]
                        and annotation["type"] == prev_annotation["type"]
                    ):
                        # find the actual output to find out how the annotation should be merged
                        output = [
                            x
                            for x in outputs
                            if x["setup"]["name"] == annotation_record["setup"]
                            and x["model"] == annotation_record["model"]
                        ][0]["generated"]

                        merged_text = output[
                            prev_annotation["start"] : annotation["start"]
                            + len(annotation["text"])
                        ]

                        prev_annotation["text"] = merged_text
                        prev_annotation["words"] += annotation["words"]

                        print(f"{prev_annotation['text']}")
                    else:
                        new_annotations.append(annotation)

            annotation_record["annotations"] = new_annotations
            new_lines.append(json.dumps(annotation_record))

    with open(jsonl_file, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

print(f"missing: {is_missing}")
print(f"off_topic: {is_off_topic}")
