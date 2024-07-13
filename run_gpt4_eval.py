#!/usr/bin/env python3

from evaluation.evaluate import GPT4Metric, Llama3Metric
import traceback

# log to file
import logging
from pathlib import Path

# logging is set by the processor, here we want to add a file handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(f"evaluation.log", mode="a")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":
    datasets = ["ice_hockey", "gsmarena", "openweather", "owid", "wikidata"]
    models = ["zephyr", "mistral", "llama2", "gpt-3.5"]
    # splits = ["dev", "test"]
    splits = ["test"]

    # this will only evaluate 3 examples from each domain
    # debug = True
    debug = False

    e = GPT4Metric()
    # e = Llama3Metric()

    for split in splits:
        for model in models:
            for dataset in datasets:
                try:
                    e.run(
                        model_name=model,
                        dataset_name=dataset,
                        split=split,
                        setup_name="direct",
                        base_path="data/quintd-1",
                        debug=debug,
                    )
                except Exception as e:
                    print(traceback.format_exc())

                    logging.error(f"Exception: {e}")
                    logging.error(f"Dataset: {dataset}, Model: {model}")
                    logging.error(traceback.format_exc())
                    logging.error("")
