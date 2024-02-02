# quintd
Code for the paper *Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation*.

## Quick pointers

- [data](data/) - **Quintd** data collection framework.
    - [generate_dataset.py](generate_dataset.py) - A script for generating a new dataset & replicating the collection of Quintd-1.
- [data/quintd-1](data/quintd-1) - Resources for the **Quintd-1** dataset.
    - [annotations](data/quintd-1/annotation) - Error annotations (GPT-4 / human).
    - [data](data/quintd-1/data) - Data inputs.
    - [outputs](data/quintd-1/outputs) - Generated model outputs.
- [run_experiments.py](run_experiments.py) - A wrapper code for running **text generation**.
- [run_gpt4_eval.py](run_experiments.py) - A wrapper code for running **GPT-4 evaluation**.

## Preliminaries
The code is tested with Python 3.10. Make sure to install the required packages first:
```bash
pip install -r requirements.txt
```

## Data Collection with Quintd
Generating a new dataset with Quintd:
```bash
python data/generate_dataset.py -d [DOMAIN] -n [EXAMPLES] -r [SEED] -o [OUT_DIR]
```

A basic setting which will generate a small dataset (10 examples per domain) with the random seed 7331:
```
SEED=7331
NUM_EXAMPLES=10

python data/generate_dataset.py -n $NUM_EXAMPLES -r $SEED
```

The dataset (Quintd-1) used for the experiments in the paper is available in [data/quintd-1](data/quintd-1).

The following code will try to replicate the data collection for Quintd-1 (up to the difference in API responses):
```
python data/generate_dataset.py --replicate
```

## Data-to-Text Generation
Data-to-text generation requires having access to the [text-generation-webui](https://github.com/oobabooga/text-generation-webui/tree/4440f87722ca9ae81e9d6123ed4b265ca2d4dae6) API.

For generating outputs for a particular domain, model, and setup:
```bash
python model/generate.py -d [DOMAIN] -s [SETUP] -m [MODEL] -p [DATASET_PATH] -a [API_URL]
```

Example:
```bash
python model/generate.py -d ice_hockey -s default -m zephyr -p data/quintd-1 -a $TG_WEBUI_API_URL 
```
You can also run the experiments with a single command:
```bash
python run_experiments.py
```
The generated outputs for Quintd-1 are available in [data/quintd-1/outputs](data/quintd-1/outputs).

## GPT-4 Error Annotation
Error annotation using GPT-4 requires access to the [OpenAI API](https://platform.openai.com/docs/api-reference).

For generating outputs for a particular domain, model, and setup:
```bash
python evaluation/evaluate.py -d [DOMAIN] -s [SETUP] -m [MODEL] -p [DATASET_PATH]
```
Example:
```bash
python evaluation/evaluate.py -d ice_hockey -s direct -m zephyr -p data/quintd-1
```
You can also run the evaluation with a single command:
```bash
python run_gpt4_eval.py
```
The error annotations for Quintd-1 are available in [data/quintd-1/annotations](data/quintd-1/annotations).