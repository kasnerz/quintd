#!/usr/bin/env python3

import requests
import argparse
import os
import json
import datetime
import pandas as pd
import time
import random
import logging
import coloredlogs
from tqdm import tqdm

coloredlogs.install(level="INFO", fmt="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_weather_from_api(location, api_key):
    # OpenWeather API URL
    base_url = f"https://api.openweathermap.org/data/2.5/forecast"
    lat, lon = location["Coordinates"].split(", ")

    # Construct the API request URL
    params = {
        "lat": lat,
        "lon": lon,
        "units": "metric",
        "mode": "json",
        "appid": api_key,
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        j = response.json()
        return j
    else:
        print("Error fetching weather data. Status code:", response.status_code)


def save_cities(seed, n_examples, out_dir, cities_path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cities_all_file = os.path.join(
        dir_path, f"geonames-all-cities-with-a-population-1000.csv"
    )

    if not os.path.exists(cities_all_file):
        logger.info(
            "Downloading the list of cities... This may take a while. The file will be saved as geonames-all-cities-with-a-population-1000.csv"
        )
        url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/exports/csv?lang=en&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"

        response = requests.get(url)
        with open(cities_all_file, "wb") as f:
            f.write(response.content)

    df = pd.read_csv(cities_all_file, sep=";")

    random.seed(seed)
    indices = random.sample(range(len(df)), n_examples * 2)
    split_indices = {
        "dev": indices[:n_examples],
        "test": indices[n_examples : n_examples * 2],
    }
    cities_for_split = {
        "dev": [],
        "test": [],
    }
    for split, indices in split_indices.items():
        for idx in indices:
            row = df.iloc[idx]
            city = {
                "Name": row["Name"],
                "Country": row["Country name EN"],
                "Coordinates": f"{row['Coordinates']}",
            }
            cities_for_split[split].append(city)

    with open(cities_path, "w") as f:
        json.dump(cities_for_split, f, indent=4)


def generate_dataset(api_key, seed, n_examples, out_dir, extra_args, verbose=False):
    all_forecasts = {"forecasts": []}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # cities_path = os.path.join(dir_path, f"cities-{seed}.json")

    # if not os.path.exists(cities_path):
    #     save_cities(seed, n_examples, out_dir, cities_path)

    # with open(cities_path) as f:
    #     cities = json.load(f)

    import csv

    cities_path = os.path.join(dir_path, "country-capital-lat-long-population.csv")
    cities = []
    with open(cities_path, mode="r") as file:
        # we need to skip a header
        next(file)
        csv_reader = csv.reader(file)
        for lines in csv_reader:
            city = {
                "Country": lines[0],
                "Name": lines[1],
                "Coordinates": f"{lines[2]}, {lines[3]}",
            }
            cities.append(city)

    # for split in ["dev", "test"]:
    # logger.info(f"Downloading forecasts for the {split} split")

    os.makedirs(os.path.join(out_dir, "cities"), exist_ok=True)

    with tqdm(total=n_examples) as pbar:
        # for city in cities[split]:
        for city in cities[:n_examples]:
            name = city["Name"]
            country = city["Country"]
            if verbose:
                logger.info(f"Downloading forecast for {name}")
            r = get_weather_from_api(city, api_key=api_key)
            all_forecasts["forecasts"].append(r)

            pbar.update(1)

            out_filename = os.path.join(out_dir, "cities", f"{name}-{country}.json")
            with open(out_filename, "w") as f:
                json.dump(all_forecasts, f, indent=4)

    out_filename = os.path.join(out_dir, f"all.json")
    with open(out_filename, "w") as f:
        json.dump(all_forecasts, f, indent=4)
