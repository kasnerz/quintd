#!/usr/bin/env python3

import requests
import json
import datetime
import time
import os
import sys
import random
import logging
import coloredlogs

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_dataset(api_key, seed, n_examples, out_dir, extra_args, verbose=False):
    splits = {
        "dev": extra_args["ice_hockey_dev_date"],
        "test": extra_args["ice_hockey_test_date"],
    }

    for split, date in splits.items():
        url = f"https://icehockeyapi.p.rapidapi.com/api/ice-hockey/matches/{date}"

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "icehockeyapi.p.rapidapi.com",
        }

        response = requests.get(url, headers=headers)
        j = response.json()

        events = j["events"]
        events = [
            e
            for e in events
            if (
                e["awayScore"]
                and e["homeScore"]
                and e["finalResultOnly"] is False
                and e["status"]["type"] == "finished"
            )
        ]
        logger.info(f"Downloading ice_hockey split {split}")
        logger.info(f"Date: {date}")
        logger.info(f"Retrieved games: {len(j['events'])}")
        logger.info(f"Games after filtering: {len(events)}")

        if len(events) < n_examples:
            logger.error(
                f"Not enough ice_hockey games available for the date {date} and the required number of examples: {n_examples}. Please choose a different date, either by specifying a different random seed, or by specifying the date manually. You can also consider manually combining examples from multiple dates."
            )
            return

        random.seed(seed)
        random.shuffle(events)
        events = events[:n_examples]

        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump(events, f, indent=4)
