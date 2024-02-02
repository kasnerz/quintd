#!/usr/bin/env python3
import requests
import json

# AutoTokenizer
from transformers import AutoTokenizer
from openai import OpenAI

import logging

logger = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, model_name, api_url, load_args=None):
        self.model_name = model_name
        self.API_URL = api_url
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model_load()

    def get_load_args(self):
        return {
            "loader": "AutoGPTQ",
            "bf16": True,
            "load_in_8bit": False,
            "groupsize": 0,
            "wbits": 0,
        }

    def model_load(self):
        logger.info(f"Loading model {self.model_name}")
        model_path = self.model_name.replace("/", "_")

        ret = self.model_api(
            {"action": "load", "model_name": model_path, "args": self.get_load_args()}
        )
        # logger.info(ret)
        self.model_info()
        return ret

    def model_api(self, request):
        response = requests.post(f"{self.API_URL}/v1/model", json=request)
        return response.json()

    def model_info(self):
        response = self.model_api({"action": "info"})
        res = response["result"]

        basic_settings = ["truncation_length", "instruction_template"]
        logger.info(f"Model: {res['model_name']}")

        for setting in basic_settings:
            logger.info(f"{setting} = {res['shared.settings'][setting]}")

        return res

    def tokenize(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="np")
        return inputs

    def normalize(self, text, start_with):
        text = text.strip()

        if text.startswith(start_with):
            text = text[len(start_with) :]

        if text.endswith('"'):
            text = text[:-1]

        if not text.endswith("."):
            text += "."

        return text

    def generate(self, prompt, params, start_with=""):
        headers = {
            "Content-Type": "application/json",
        }

        if start_with:
            arr = [prompt, start_with]
            history = {"internal": [arr], "visible": [arr]}
            request = {
                "user_input": prompt,
                "history": history,
                "stopping_strings": ["[INST]"],
                "_continue": True,
                **params,
            }
        else:
            request = {"user_input": prompt, "mode": "instruct", **params}

        response = requests.post(
            self.API_URL + "/v1/chat", headers=headers, data=json.dumps(request)
        )
        j = response.json()

        try:
            response = j["results"][0]["history"]["internal"][0][1]
            response = self.normalize(response, start_with)

            return response
        except KeyError:
            logger.error("KeyError:")
            logger.error(j)
            return None


class MistralInstructModel(LanguageModel):
    def __init__(self, api_url):
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", api_url=api_url)


class Llama2Model(LanguageModel):
    def __init__(self, api_url):
        super().__init__("togethercomputer/Llama-2-7B-32K-Instruct", api_url=api_url)


class ZephyrBetaModel(LanguageModel):
    def __init__(self, api_url):
        super().__init__("HuggingFaceH4/zephyr-7b-beta", api_url=api_url)


class ChatGPTModel(LanguageModel):
    def __init__(self, api_url):
        self.client = OpenAI()
        self.model_name = "gpt-3.5-turbo-1106"
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def generate(self, prompt, params, start_with=""):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": start_with},
                ],
                seed=0,
                temperature=0,
                max_tokens=params["max_new_tokens"],
                stop=["\n\n"],
            )
            response = response.choices[0].message.content

            response = self.normalize(response, start_with)
            return response
        except Exception as e:
            logger.error(e)
            return ""

    def model_load(self):
        logger.info("ChatGPT ready")

    def model_api(self, request):
        return None

    def model_info(self):
        logger.info(self.model_name)
