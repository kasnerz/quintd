#!/usr/bin/env python3
from owid import catalog
import os
from pandas import DataFrame
import json
import numpy as np
import time
import shutil
import random
import pandas as pd
from pathlib import Path

import logging
import coloredlogs
from tqdm import tqdm

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def preprocess(metadata_dir, fetched_out_dir, seed):
    # generate a CSV file with metadata, prune
    data = []

    for filename in Path(fetched_out_dir).iterdir():
        column = filename.stem.split("-")[0]
        country_name = filename.stem.split("-")[1]

        metadata_file = Path(f"{metadata_dir}/{column}.json")

        with open(metadata_file) as f:
            metadata = json.load(f)

        with open(filename) as f:
            df = pd.read_csv(f)

            # keep the length below ~8k tokens to prevent OOM errors
            approx_tokens = len(df.to_csv(index=False))
            downsample_factor = approx_tokens // 8000 + 1

            df = df.iloc[::downsample_factor, :]
            df_csv = df.to_csv(index=False)

            csv_str = (
                f"# country: {country_name}\n# title: {metadata['title']}\n# description: {metadata['description']}\n# unit: {metadata['unit']}\n"
                + df_csv
            )
            data.append((filename, csv_str))

    random.seed(seed)
    random.shuffle(data)

    return data


def extract_covid():
    logger.info("Extracting COVID data")

    all_data = []
    df = catalog.find("covid")
    table = df.iloc[0].load()
    countries = table.groupby("location")
    columns = [
        "new_cases_smoothed_per_million",
        "new_tests_smoothed_per_thousand",
        "people_vaccinated_per_hundred",
        "reproduction_rate",
        "positive_rate",
    ]

    for country, data in countries:
        for column in columns:
            if data[column].isna().all():
                # print(f"Skipping {column} for {country} - all values are NaN")
                continue
            new_df = DataFrame(columns=["date", "value"])
            new_df["date"] = data[column].index.get_level_values(1)
            new_df["value"] = data[column].values

            # remove nan values and <class 'pandas._libs.missing.NAType'>
            new_df = new_df.dropna()

            # if no data is left, skip
            if len(new_df) == 0:
                continue

            all_data.append({"df": new_df, "country": country, "column": column})

    return all_data


def extract_expectancy():
    logger.info("Extracting life expectancy data")

    all_data = []
    df = catalog.find("life_expectancy")
    table = df.loc[df["table"] == "life_expectancy"].iloc[0].load()
    countries = table.groupby("country")

    column = "life_expectancy_0"

    for i, (country, data) in enumerate(countries):
        if data[column].isna().all():
            print(f"Skipping {column} for {country} - all values are NaN")
            continue

        new_df = DataFrame(columns=["date", "value"])
        new_df["date"] = data[column].index.get_level_values(1)
        new_df["value"] = data[column].values

        # remove nan values and <class 'pandas._libs.missing.NAType'>
        new_df = new_df.dropna()

        # if no data is left, skip
        if len(new_df) == 0:
            continue

        all_data.append({"df": new_df, "country": country, "column": column})

    return all_data


def generate_dataset(api_key, seed, n_examples, out_dir, extra_args, verbose=False):
    all_csv = []
    all_csv += extract_covid()
    all_csv += extract_expectancy()

    # logger.info(f"Total number of CSVs: {len(all_csv)}")
    random.seed(seed)
    indices = random.sample(range(len(all_csv)), n_examples * 2)
    split_indices = {
        "dev": indices[:n_examples],
        "test": indices[n_examples : n_examples * 2],
    }
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for split, indices in split_indices.items():
        logger.info(f"Generating the {split} split")

        fetched_out_dir = os.path.join(dir_path, "fetched", split)
        metadata_dir = os.path.join(dir_path, "metadata")
        split_out_dir = os.path.join(out_dir, split)

        # if the fetched data already exists, remove the directory
        if os.path.exists(fetched_out_dir):
            shutil.rmtree(fetched_out_dir)

        os.makedirs(fetched_out_dir, exist_ok=True)
        os.makedirs(split_out_dir, exist_ok=True)

        for idx in indices:
            row = all_csv[idx]
            country = row["country"]
            column = row["column"]
            df = row["df"]

            with open(
                os.path.join(fetched_out_dir, f"{column}-{country}.csv"), "w"
            ) as f:
                df.to_csv(f, index=False)

        data = preprocess(
            metadata_dir=metadata_dir, fetched_out_dir=fetched_out_dir, seed=seed
        )

        for i, (filename, csv_str) in enumerate(data):
            filename = filename.name
            new_filename = f"{i}-{filename}"
            with open(os.path.join(split_out_dir, new_filename), "w") as f:
                f.write(csv_str)
