#!/usr/bin/env python3

import json
import random
from collections import defaultdict
from pprint import pprint as pp
import argparse
import yaml
import os
import logging
import coloredlogs
import datetime

import api.openweather.openweather as openweather
import api.gsmarena.gsmarena as gsmarena
import api.ice_hockey.ice_hockey as ice_hockey
import api.owid.main as owid
import api.wikidata.wikidata as wikidata

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_api_key(api_keys, api_name):
    key = f"{api_name.upper()}_API_KEY"
    if api_keys.get(key, None) == "":
        logger.warning(
            f"API key for {api_name} missing. Please add the API key to quintd/data_collection/api_keys.yaml"
        )
    api_key = api_keys.get(key) or os.environ.get(key)
    return api_key


if __name__ == "__main__":
    domains = {
        "ice_hockey": ice_hockey,
        "gsmarena": gsmarena,
        "openweather": openweather,
        "owid": owid,
        "wikidata": wikidata,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--domains",
        nargs="+",
        type=str,
        choices=domains.keys(),
        default=domains.keys(),
        help="Domains to generate the dataset for (default: all domains).",
    )
    parser.add_argument(
        "-n",
        "--examples",
        type=int,
        default=100,
        help="Number of examples to generate per domain in each split.",
    )
    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        default=0,
        help="Random seed. Each random seed will generate a different dataset.",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for the generated dataset.",
    )
    parser.add_argument(
        "-v", "--verbose", type=str, default=None, help="Log extra information."
    )
    parser.add_argument(
        "--gsmarena_full",
        action="store_true",
        help="Attempt downloading products directly from GSMArena. This may run into limitations of the website. If not specified, the products will be selected from the pre-downloaded list of products.",
    )
    parser.add_argument(
        "--ice_hockey_date",
        type=str,
        default=None,
        help="Matches for ice_hockey are downloaded for a specific date. The date will be selected from a range based on the random seed. However, you can also specify the date for the `dev` set manually. The format in the format DD/MM/YYYY. The test set will be generated for the next day after the dev set.",
    )
    parser.add_argument(
        "--replicate",
        action="store_true",
        default=False,
        help="Replicate the collection of the Quintd-1 dataset. Note that the replication may not be exact as the API responses may differ.",
    )
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    api_keys = yaml.load(
        open(os.path.join(dir_path, "api_keys.yaml"), "r"), Loader=yaml.FullLoader
    )

    if args.replicate:
        raise NotImplementedError(
            "Replication of the Quintd-1 dataset is not implemented yet."
        )
        seeds = {
            "ice_hockey": 42,
            "gsmarena": 42,
            "openweather": 0,
            "owid": 0,
            "wikidata": 42,
        }
        ice_hockey_dev_date = "27/11/2023"
        ice_hockey_test_date = "29/11/2023"
        examples = 100

    if args.examples > 100:
        logger.warning(
            "Generating more than 100 examples per domain is not recommended as some APIs may impose fees. Please make sure you are accomodated with the API policies. Are you sure you want to continue? (y/n)"
        )
        if input() != "y":
            exit()

    if args.out_dir is None:
        out_dir = os.path.join(dir_path, f"quintd-custom-{args.seed}")

    if args.ice_hockey_date is None:
        # date is not specified explicitly, so we generate a random date between Oct. 7, 2022 to Apr. 14, 2023 (NHL season)
        random.seed(args.seed)
        ice_hockey_dev_date = (
            datetime.date(2022, 10, 7) + datetime.timedelta(days=random.randint(0, 189))
        ).strftime("%d/%m/%Y")
    else:
        # date is specified explicitly
        ice_hockey_dev_date = args.ice_hockey_date

    # test_date is one day after dev_date (to have enough matches for the split)
    ice_hockey_test_date = (
        datetime.datetime.strptime(ice_hockey_dev_date, "%d/%m/%Y").date()
        + datetime.timedelta(days=1)
    ).strftime("%d/%m/%Y")

    extra_args = {
        "gsmarena_full": args.gsmarena_full,
        "ice_hockey_dev_date": ice_hockey_dev_date,
        "ice_hockey_test_date": ice_hockey_test_date,
    }

    for domain in args.domains:
        module = domains[domain]
        os.makedirs(os.path.join(out_dir, domain), exist_ok=True)

        logger.info(f"Processing {domain} dataset")
        openweather_api_key = get_api_key(api_keys, domain)
        openweather_out_dir = os.path.join(out_dir, domain)

        module.generate_dataset(
            api_key=openweather_api_key,
            seed=args.seed,
            n_examples=args.examples,
            out_dir=openweather_out_dir,
            extra_args=extra_args,
            verbose=args.verbose,
        )
