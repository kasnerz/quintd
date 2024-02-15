#!/usr/bin/env python3

import re
import glob
import json
import random
import os
import argparse
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import OpenWeather, IceHockey, GSMArena, Wikidata, OurWorldInData

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "quintd-1")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

dataset_dict = {
    "openweather": OpenWeather,
    "ice_hockey": IceHockey,
    "gsmarena": GSMArena,
    "wikidata": Wikidata,
    "owid": OurWorldInData,
}

error_mapping = {
    0: "Incorrect",
    1: "Not checkable",
    2: "Misleading",
    3: "Other",
    4: "All categories",
}


def load_dataset(dataset_name):
    return dataset_dict[dataset_name](base_path=DATASET_DIR)


def generate_annotation_index():
    """
    All annotations as a pandas dataframe, each error is a separate row
    """
    annotation_index = []

    for source in ["gpt-4", "human"]:
        jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, source, "*.jsonl"))

        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for line in f:
                    annotation_record = json.loads(line)
                    annotations = annotation_record["annotations"]

                    for annotation in annotations:
                        # prefix all the keys in annotation with "annotation_"
                        error_record = {
                            f"annotation_{k}": v for k, v in annotation.items()
                        }
                        # append all the fields from the annotation record
                        error_record.update(annotation_record)
                        error_record.pop("annotations")
                        # add the source
                        error_record["source"] = source
                        annotation_index.append(error_record)

    df = pd.DataFrame(annotation_index)

    return df


def generate_human_annotation_index():
    annotation_index = []
    jsonl_files = Path(ANNOTATIONS_DIR) / "human_corr"
    jsonl_files = sorted(list(jsonl_files.rglob("*.jsonl")))
    human_ctr = 0
    current_table_idx = None

    for jsonl_file in jsonl_files:
        table_idx = int(jsonl_file.stem.split("-")[0])

        if table_idx != current_table_idx:
            human_ctr = 0
            current_table_idx = table_idx
        else:
            human_ctr += 1

        with open(jsonl_file) as f:
            for line in f:
                annotation_record = json.loads(line)
                annotations = annotation_record["annotations"]

                for annotation in annotations:
                    # prefix all the keys in annotation with "annotation_"
                    error_record = {f"annotation_{k}": v for k, v in annotation.items()}
                    # append all the fields from the annotation record
                    error_record.update(annotation_record)
                    error_record.pop("annotations")
                    # add the source
                    # error_record["source"] = "human-" + str(human_ctr)

                    # this is very stupid but it allows us to compute easily the correlations between human annotators with the existing code: we simply consider the second human annotator as the "gpt-4" annotator
                    if human_ctr == 0:
                        error_record["source"] = "human"
                    elif human_ctr == 1:
                        error_record["source"] = "gpt-4"
                    else:
                        continue

                    annotation_index.append(error_record)

    df = pd.DataFrame(annotation_index)
    return df


def generate_annotation_key_value_index():
    """
    All annotations as a dictionary, keys are tuples of (dataset, split, table_idx, model, setup)
    """
    annotations = defaultdict(list)

    for source in ["gpt-4", "human"]:
        jsonl_files = glob.glob(os.path.join(ANNOTATIONS_DIR, source, "*.jsonl"))

        for jsonl_file in jsonl_files:
            with open(jsonl_file) as f:
                for line in f:
                    annotation = json.loads(line)
                    key = (
                        annotation["dataset"],
                        annotation["split"],
                        annotation["table_idx"],
                        annotation["model"],
                        annotation["setup"],
                    )
                    annotations[key].append(annotation)

    return annotations


def generate_human_annotation_key_value_index():
    """
    All annotations as a dictionary, keys are tuples of (dataset, split, table_idx, model, setup)
    """
    annotations = defaultdict(list)
    jsonl_files = Path(ANNOTATIONS_DIR) / "human_corr"
    jsonl_files = sorted(list(jsonl_files.rglob("*.jsonl")))
    human_ctr = 0
    current_table_idx = None

    for jsonl_file in jsonl_files:
        table_idx = int(jsonl_file.stem.split("-")[0])

        if table_idx != current_table_idx:
            human_ctr = 0
            current_table_idx = table_idx
        else:
            human_ctr += 1

        with open(jsonl_file) as f:
            for line in f:
                annotation = json.loads(line)
                key = (
                    annotation["dataset"],
                    annotation["split"],
                    annotation["table_idx"],
                    annotation["model"],
                    annotation["setup"],
                )

                # hotfix to be able use existing code for human correlation, see also generate_human_annotation_index
                if human_ctr > 1:
                    breakpoint()
                    continue
                elif human_ctr == 1:
                    annotation["annotator_id"] = "gpt-4"

                annotations[key].append(annotation)

    return annotations


def generate_model_outputs_latex(dataset_name, table_idx):
    pd.set_option("display.max_colwidth", None)

    annotations = generate_annotation_key_value_index()

    dataset = load_dataset(dataset_name)
    split = "test"

    generated_outputs = dataset.get_generated_outputs(split, table_idx)

    df = []

    for generated_output in generated_outputs:
        if generated_output["setup"]["name"] != "direct":
            continue

        model = generated_output["model"]
        key = (dataset.name, split, table_idx, model, "direct")

        annotation_fullset = annotations.get(key, [])

        # get human annotations
        annotations_example = {
            "Human": [
                x
                for x in annotation_fullset
                if not x["annotator_id"].startswith("gpt-4")
            ][0],
            "GPT-4": [
                x for x in annotation_fullset if x["annotator_id"].startswith("gpt-4")
            ][0],
        }

        entry = {
            "Model": model,
        }
        for source in annotations_example.keys():
            output = generated_output["generated"]
            annotation_set = annotations_example[source]
            annotation_list = annotation_set["annotations"]

            # sort the annotations by start position
            annotation_list = sorted(annotation_list, key=lambda x: x["start"])

            shift = 0
            previous_end = 0

            for annotation in annotation_list:
                start = annotation["start"] + shift
                end = start + len(annotation["text"])

                if end < previous_end:
                    continue

                previous_end = end

                error_str = error_mapping[annotation["type"]].lower().replace(" ", "")

                highlighted_text = f"\\ctext{{{error_str}}}{{{output[start:end]}}}"
                output = output[:start] + highlighted_text + output[end:]
                shift += len(highlighted_text) - len(annotation["text"])

            entry[source] = output

        df.append(entry)

    df = pd.DataFrame(df)
    # sort in order : "llama2", "mistral", "zephyr", "gpt-3.5"
    df = df.sort_values(
        by=["Model"],
        key=lambda x: x.map({"llama2": 0, "mistral": 1, "zephyr": 2, "gpt-3.5": 3}),
    )
    df = df.reset_index(drop=True)

    # title case the model names
    df["Model"] = df["Model"].str.title()
    df["Model"] = df["Model"].str.replace("Gpt-3.5", "GPT-3.5")

    styler = df.style
    styler.hide(axis="index")
    styler.applymap(lambda v: "font-weight: bold;", subset=["Model"])
    styler.applymap_index(lambda v: "font-weight: bold;", axis="columns")
    styler.format(escape="latex")
    table = styler.to_latex(
        convert_css=True, hrules=True, column_format="lp{6.5cm}p{6.5cm}"
    )

    # table = table.replace("%", "\\%")
    table = table.replace("°", "\\textdegree{}")

    table = re.sub(
        r"\\textbackslash ctext\\{(\w+)\\}\\{(.*?)\\}", r"\\ctext{\1}{\2}", table
    )

    table = table.replace(
        "\\bfseries Human & GPT-4",
        "\\textbf{Human annotations} (\\humanmetric) & \\textbf{GPT-4 annotations} (\\gptmetric)",
    )

    # print(table)

    with open(f"output_{dataset_name}.tex", "w") as f:
        f.write(table)


def generate_token_and_example_level_table(annotations):
    token_list = []
    example_list = []
    split = "test"
    for dataset in dataset_dict.keys():
        dataset = load_dataset(dataset)

        print(f"Processing {dataset.name}")

        for table_idx in range(100):
            generated_outputs = dataset.get_generated_outputs(split, table_idx)

            for generated_output in generated_outputs:
                model = generated_output["model"]
                setup = generated_output["setup"]["name"]

                if setup != "direct":
                    continue

                key = (dataset.name, split, table_idx, model, setup)

                annotation_fullset = annotations.get(key, [])

                example = {
                    "errors_human": {x: 0 for x in range(4)},
                    "errors_gpt-4": {x: 0 for x in range(4)},
                    "dataset": dataset.name,
                    "model": model,
                    "setup": setup,
                    "split": split,
                    "table_idx": table_idx,
                }

                tokens = [
                    {
                        "token": m.group(0),
                        "start": m.start(),
                        "end": m.end() - 1,
                        "error_human": None,
                        "error_gpt-4": None,
                        "dataset": dataset.name,
                        "model": model,
                        "setup": setup,
                        "split": split,
                        "table_idx": table_idx,
                    }
                    for m in re.finditer(r"\S+", generated_output["generated"])
                ]

                # sort annotation_fullset by annotator_id
                annotation_fullset = sorted(
                    annotation_fullset, key=lambda x: x["annotator_id"]
                )

                if len(annotation_fullset) > 2:
                    print(
                        f"Warning: more than 2 annotations for {dataset.name} {split} {table_idx} {model} {setup}"
                    )

                for annotation_set in annotation_fullset:
                    annotation_list = annotation_set["annotations"]
                    source = (
                        "gpt-4"
                        if annotation_set["annotator_id"].startswith("gpt-4")
                        else "human"
                    )

                    for annotation in annotation_list:
                        start = annotation["start"]
                        end = start + len(annotation["text"])

                        if annotation["type"] not in range(4):
                            print(
                                f"Invalid annotation type {annotation['type']} in {annotation}"
                            )
                            continue

                        example[f"errors_{source}"][annotation["type"]] += 1

                        # mark all tokens that are at least partially inside the annotation span as having the error type
                        for token in tokens:
                            if (
                                (
                                    token["start"] >= start and token["start"] <= end
                                )  # token starts inside the annotation span
                                or (
                                    token["end"] >= start and token["end"] <= end
                                )  # token ends inside the annotation span
                                or (
                                    token["start"] <= start and token["end"] >= end
                                )  # token fully contains the annotation span
                                or (token["start"] >= start and token["end"] <= end)
                            ):  # annotation fully contains the token
                                token[f"error_{source}"] = int(annotation["type"])

                token_list.extend(tokens)
                example_list.append(example)

    df_tokens = pd.DataFrame(token_list)
    df_examples = pd.DataFrame(example_list)

    return df_tokens, df_examples


def generate_human_corr(df_tokens, df_examples, df_domain):
    pass


def generate_metric_corr(df_tokens, df_examples, df_domain):
    tokens_total = len(df_tokens)

    errors_human = df_tokens["error_human"].value_counts() / tokens_total
    errors_gpt4 = df_tokens["error_gpt-4"].value_counts() / tokens_total

    # print a latex table showing a percentage of errors for each category
    # categories are rows, and columns are human / gpt-4
    # also print a total percentage of errors as the last row

    errors_both = df_tokens[df_tokens["error_human"] == df_tokens["error_gpt-4"]]
    errors_both = errors_both["error_human"].value_counts() / tokens_total

    df_errors = pd.DataFrame(
        {
            "GPT-4": errors_gpt4,
            "Human": errors_human,
            "Both": errors_both,
        }
    )
    df_errors = df_errors.rename(index=error_mapping)

    # add the total number of errors
    df_errors.loc["Total"] = df_errors.sum()
    df_errors = df_errors.round(3)

    print(df_errors.to_latex())

    # token-level Pearson correlation
    human_list_all = []
    gpt4_list_all = []

    for err_cat in error_mapping.keys():
        human_list = [1 if x else 0 for x in df_tokens["error_human"] == err_cat]
        gpt4_list = [1 if x else 0 for x in df_tokens["error_gpt-4"] == err_cat]

        human_list_all.extend(human_list)
        gpt4_list_all.extend(gpt4_list)

    coeff, p = pearsonr(human_list_all, gpt4_list_all)
    print(f"Token-level Pearson correlation: {coeff:.2f} (p={p:.2f})")

    # example-level Pearson correlation
    human_list_all = []
    gpt4_list_all = []

    for i, example in df_examples.iterrows():
        human_list_all.extend(list(example["errors_human"].values()))
        gpt4_list_all.extend(list(example["errors_gpt-4"].values()))

    coeff, p = pearsonr(human_list_all, gpt4_list_all)
    print(f"Example-level Pearson correlation: {coeff:.2f} (p={p:.2f})")

    # domain-level Pearson correlation

    # columns are indexed by tuples ('Incorrect', 'gpt-4')","('Incorrect', 'human')","('Not checkable', 'gpt-4'), etc.
    df_human_columns = [
        x for x in df_domain.columns if type(x) is tuple and x[1] == "human"
    ]
    df_gpt4_columns = [
        x for x in df_domain.columns if type(x) is tuple and x[1] == "gpt-4"
    ]

    df_human = df_domain[df_human_columns].copy()
    df_gpt4 = df_domain[df_gpt4_columns].copy()

    vals_human = df_human.values.flatten()
    vals_gpt4 = df_gpt4.values.flatten()

    coeff, p = pearsonr(vals_human, vals_gpt4)
    print(f"Domain-level Pearson correlation: {coeff:.2f} (p={p:.2f})")

    # errors_human_only = df_tokens[(df_tokens["error_human"].notnull()) & (df_tokens["error_gpt-4"].isnull())]
    # errors_gpt4_only = df_tokens[(df_tokens["error_human"].isnull()) & (df_tokens["error_gpt-4"].notnull())]
    # errors_none = df_tokens[(df_tokens["error_human"].isnull()) & (df_tokens["error_gpt-4"].isnull())]

    # print(f"Human only: {len(errors_human_only)} ({len(errors_human_only) * 100 / tokens_total:.2f}%)")
    # print(f"GPT-4 only: {len(errors_gpt4_only)} ({len(errors_gpt4_only) * 100 / tokens_total:.2f}%)")
    # print(f"Both: {len(errors_both)} ({len(errors_both) * 100 / tokens_total:.2f}%)")
    # print(f"None: {len(errors_none)} ({len(errors_none) * 100 / tokens_total:.2f}%)")


def generate_length_table():
    lengths_df = []
    split = "test"

    for dataset in dataset_dict.keys():
        dataset = load_dataset(dataset)

        print(f"Processing {dataset.name}")

        for table_idx in range(100):
            generated_outputs = dataset.get_generated_outputs(split, table_idx)

            for generated_output in generated_outputs:
                model = generated_output["model"]
                setup = generated_output["setup"]["name"]

                if setup != "direct":
                    continue

                length_record = {
                    "dataset": dataset.name,
                    "table_idx": table_idx,
                    "model": model,
                    "length": len(generated_output["generated"].split()),
                }

                lengths_df.append(length_record)

    lengths_df = pd.DataFrame(lengths_df)
    lengths_df = lengths_df.groupby(["model", "dataset"]).mean().reset_index()

    # remove the `table_idx` column from the lengths dataframe
    lengths_df = lengths_df.drop(columns=["table_idx"])

    return lengths_df


def normalize_names(df):
    df["model"] = pd.Categorical(
        df["model"], ["llama2", "mistral", "zephyr", "gpt-3.5"]
    )

    df["dataset"] = pd.Categorical(
        df["dataset"], ["openweather", "gsmarena", "ice_hockey", "owid", "wikidata"]
    )

    df["annotation_type"] = df["annotation_type"].map(error_mapping)

    return df


def generate_tables_ex_err_ratio(df):
    split = "test"
    orig_df = df[df["split"] == split]

    assert sorted(list(orig_df.table_idx.unique())) == list(range(100))

    #  delete any record whose annotation_type is not in [0, 1, 2, 3]
    df = orig_df[orig_df["annotation_type"].isin([0, 1, 2, 3])]

    df_percat = (
        df.groupby(["source", "model", "annotation_type", "dataset"])["table_idx"]
        .nunique()
        .reset_index()
    )

    df_allcat = (
        df.groupby(["source", "model", "dataset"])["table_idx"].nunique().reset_index()
    )

    # merge the dataframes
    # create a new `annotation_type` column "All categories" for `df_allcat`
    df_allcat["annotation_type"] = 4
    df = pd.concat([df_percat, df_allcat])

    df = normalize_names(df)

    # make the "All categories" annotation type the last one
    df = df.pivot_table(
        values="table_idx",
        index=["model", "dataset"],
        columns=["annotation_type", "source"],
        aggfunc="first",
    )

    order = error_mapping.values()
    df = df.reindex(columns=order, level=0)

    df = df.fillna(0)
    df = df / 100
    df = df.round(2)
    print(df.to_csv())

    # print also an average for each model across all datasets
    df = df.groupby(["model"]).mean().reset_index()
    df = df.round(2)

    print(df.to_csv())


def generate_tables_avg_errors(df):
    split = "test"

    orig_df = df[df["split"] == split]

    # disable for now: we don't have 100 examples for human correlations
    # assert sorted(list(orig_df.table_idx.unique())) == list(range(100))

    #  delete any record whose annotation_type is not in [0, 1, 2, 3]
    df = orig_df[orig_df["annotation_type"].isin([0, 1, 2, 3])]

    nr_examples = len(df["table_idx"].unique())
    # print(f"Number of examples: {nr_examples}")

    # # aggregate annotations by model, dataset, and error type
    df_percat = (
        df.groupby(["model", "dataset", "source", "annotation_type"])
        .size()
        .reset_index(name="count")
    )
    df_allcat = (
        df.groupby(["model", "dataset", "source"]).size().reset_index(name="count")
    )

    # merge the dataframes
    # create a new `annotation_type` column "All categories" for `df_allcat`
    df_allcat["annotation_type"] = 4
    df = pd.concat([df_percat, df_allcat])

    df = normalize_names(df)

    df = df.pivot_table(
        values="annotation_type",
        index=["model", "dataset"],
        columns=["annotation_type", "source"],
        aggfunc="first",
    )

    order = error_mapping.values()
    df = df.reindex(columns=order, level=0)

    df = df.fillna(0)
    df = df / nr_examples
    df = df.round(2)

    lengths_df = generate_length_table()

    # append the `length` column to the main dataframe
    df = df.merge(lengths_df, on=["model", "dataset"])

    print(df.to_csv())

    # print also an average for each model across all datasets
    df_agg = df.groupby(["model"]).mean().reset_index()
    df_agg = df_agg.round(2)
    print(df_agg.to_csv())
    return df, df_agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results",
        choices=["avg_errors", "ex_err_ratio", "metric_corr", "human_corr", "outputs"],
        help="Dataset",
        required=True,
    )
    args = parser.parse_args()

    if args.results == "avg_errors":
        df = generate_annotation_index()
        generate_tables_avg_errors(df)

    elif args.results == "ex_err_ratio":
        df = generate_annotation_index()
        generate_tables_ex_err_ratio(df)

    elif args.results == "metric_corr":
        df = generate_annotation_index()
        annotations = generate_annotation_key_value_index()

        df_tokens, df_examples = generate_token_and_example_level_table(annotations)
        df_domain, df_domain_agg = generate_tables_avg_errors(df)
        generate_metric_corr(df_tokens, df_examples, df_domain)

    elif args.results == "human_corr":
        df = generate_human_annotation_index()
        human_annotations = generate_human_annotation_key_value_index()

        df_tokens, df_examples = generate_token_and_example_level_table(
            human_annotations
        )
        df_domain, df_domain_agg = generate_tables_avg_errors(df)
        generate_metric_corr(df_tokens, df_examples, df_domain)

    elif args.results == "outputs":
        random.seed(0)
        idx = random.randint(0, 99)

        print(f"Generating tables for example #{idx}")
        for dataset_name in dataset_dict.keys():
            generate_model_outputs_latex(dataset_name, idx)
