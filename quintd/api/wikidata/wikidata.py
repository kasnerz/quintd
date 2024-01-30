#!/usr/bin/env python3

import pickle
import wikidatasets
import pandas as pd
import tqdm
import json
import os
import numpy as np
import random
import tarfile
from collections import defaultdict
import logging
import coloredlogs
import requests

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_eng_alpha(c):
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z")


def extract_subgraphs(subgraphs_per_domain):
    domains = ["companies", "countries", "films", "humans"]
    data = []

    for subdomain in domains:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dir_path, "data")
        subdir = os.path.join(data_dir, subdomain)

        if not os.path.exists(subdir):
            url = (
                f"https://graphs.telecom-paris.fr/data/WikiDataSets/{subdomain}.tar.gz"
            )
            archive_path = os.path.join(data_dir, f"{subdomain}.tar.gz")
            logger.info(f"Pre-downloading {subdomain} data into {archive_path}")

            response = requests.get(url)
            with open(archive_path, "wb") as f:
                f.write(response.content)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(data_dir)

            logger.info(f"Removing {archive_path}")
            # Remove the archive file
            os.remove(archive_path)

        df = wikidatasets.utils.load_data_labels(subdir + "/", attributes=True)
        logger.info(f"Processing {subdomain}")

        # sample `subgraphs_per_domain` unique entities
        entities = df["headLabel"].unique()

        # keep only entities beginning with a letter of English alphabet
        entities = [x for x in entities if (type(x) is str) and is_eng_alpha(x[0])]
        entities = np.random.choice(entities, size=subgraphs_per_domain, replace=False)

        for i in tqdm.tqdm(range(subgraphs_per_domain)):
            entity_triples = []
            entity = entities[i]
            min_relations = 2
            max_relations = 10
            relevant_triples = df[df["headLabel"] == entity]

            # sample relevant triples
            relevant_triples = relevant_triples.sample(
                min(max_relations, len(relevant_triples))
            )

            # keep adding triples until we reach max_relations
            for _, row in relevant_triples.iterrows():
                entity_triples.append((row["relationLabel"], row["tailLabel"]))

            if len(entity_triples) < min_relations:
                continue

            # add entity triples to data
            data.append(
                {
                    "entity": entity,
                    "properties": sorted(entity_triples, key=lambda x: x[0]),
                }
            )

    return data


def generate_dataset(api_key, seed, n_examples, out_dir, extra_args, verbose=False):
    np.random.seed(0)
    data = extract_subgraphs(subgraphs_per_domain=n_examples)

    # print histogram of triple lengths
    relation_histogram = defaultdict(int)
    for example in data:
        triples = example["properties"]
        relation_histogram[len(triples)] += 1

    # sort
    relation_histogram = sorted(relation_histogram.items(), key=lambda x: x[0])

    # print("Total number of subgraphs:", len(data))
    # print("Histogram of triple lengths:", relation_histogram)

    random.seed(seed)
    random.shuffle(data)

    examples = {
        "dev": data[:n_examples],
        "test": data[n_examples : n_examples * 2],
    }

    for split, split_data in examples.items():
        # save data
        with open(os.path.join(out_dir, f"{split}.json"), "w") as f:
            json.dump(split_data, f, indent=4, ensure_ascii=False)
