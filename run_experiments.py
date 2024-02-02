#!/usr/bin/env python3

from model.generate import Processor
import traceback
import logging
import os

# logging is set by the processor, here we want to add a file handler
logger = logging.getLogger()
fh = logging.FileHandler(f"experiments.log", mode="w")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":
    datasets = ["ice_hockey", "gsmarena", "openweather", "owid", "wikidata"]
    models = ["zephyr", "mistral", "llama2", "gpt-3.5"]
    setups = ["direct"]
    splits = ["dev", "test"]
    api_url = os.environ.get("TG_WEBUI_API_URL")

    if not api_url:
        raise ValueError(
            "Please set the TG_WEBUI_API_URL environment variable to the URL of the text-generation-webui API."
        )

    # this will only generate 3 examples from each domain
    debug = True

    for split in splits:
        for model in models:
            p = Processor(model_name=model, api_url=api_url)

            for dataset in datasets:
                for setup_name in setups:
                    try:
                        p.run(
                            dataset_name=dataset,
                            setup_name=setup_name,
                            split=split,
                            base_path="data/quintd-1-test",
                            debug=debug,
                        )
                    except Exception as e:
                        print(traceback.format_exc())

                        logging.error(f"Exception: {e}")
                        logging.error(
                            f"Dataset: {dataset}, Model: {model}, Setup: {setup_name}"
                        )
                        logging.error(traceback.format_exc())
                        logging.error("")
