import os
import json
import random
import logging
import coloredlogs

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_dataset_full(seed, n_examples, dir_path, out_dir):
    gsmarena_api_path = f"{dir_path}/gsmarena-api"

    if not os.path.exists(gsmarena_api_path):
        os.system(
            f"git clone https://github.com/nordmarin/gsmarena-api.git {gsmarena_api_path} && cd {gsmarena_api_path} && git checkout 3bf1841"
        )

    max_phones_per_brand = 10
    os.system(
        f"npm i gsmarena-api && node {dir_path}/scraper.js {out_dir} {max_phones_per_brand}"
    )


def generate_dataset(api_key, seed, n_examples, out_dir, flags, verbose=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    if flags["gsmarena_full"]:
        logger.warning(
            "Downloading products directly from GSMArena. This may take a while. If the API returns 'Too Many Requests' error, the script will retry later: please give it enough time to finish."
        )
        logger.warning(
            "NPM is required for the gsmarena-api. Please make sure you have NPM installed."
        )
        generate_dataset_full(seed, n_examples, dir_path=dir_path, out_dir=out_dir)
    else:
        with open(f"{dir_path}/phones_predownloaded.json") as f:
            j = json.load(f)

        # flatten the `deviceList`
        all_devices = [d for l in j for d in l["deviceList"] if d["details"]["name"]]

        # drop `description` and `img` keys
        for device in all_devices:
            device.pop("description", None)
            device.pop("img", None)

        random.seed(seed)
        random.shuffle(all_devices)

        ranges = {
            "dev": (0, n_examples),
            "test": (n_examples, n_examples * 2),
        }

        for split in ["dev", "test"]:
            examples = all_devices[ranges[split][0] : ranges[split][1]]

            with open(f"{out_dir}/{split}.json", "w") as f:
                json.dump(examples, f, indent=4)
