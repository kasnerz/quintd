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
            "loader": "Transformers",
            "bf16": True,
            "load_in_8bit": False,
            # "groupsize": 0,
            # "wbits": 0,
        }

    def model_load(self):
        logger.info(f"Loading model {self.model_name}")
        model_path = self.model_name.replace("/", "_")

        ret = self.model_api(
            {"action": "load", "model_name": model_path, "args": self.get_load_args()}
        )
        # logger.info(ret)
        # self.model_info()
        return ret

    def model_api(self, request):
        response = requests.post(f"{self.API_URL}/v1/internal/model/load", json=request)
        return response.json()

    # def model_info(self):
    #     response = self.model_api({"action": "info"})
    #     res = response["result"]

    #     basic_settings = ["truncation_length", "instruction_template"]
    #     logger.info(f"Model: {res['model_name']}")

    #     for setting in basic_settings:
    #         logger.info(f"{setting} = {res['shared.settings'][setting]}")

    #     return res

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
        messages = [{"role": "user", "content": prompt}]
        if start_with:
            messages.append({"role": "assistant", "content": start_with})

        data = {"messages": messages, "stop": ["[INST]"], **params}

        response = requests.post(
            f"{self.API_URL}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
            },
            json=data,
            verify=False,
        )
        try:
            output_text = response.json()["choices"][0]["message"]["content"]
            output_text = self.normalize(output_text, start_with)
            return output_text
        except:
            logger.error(data)
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
                max_tokens=params["max_tokens"],
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
