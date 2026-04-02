#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        rea_fuser.py
@Author:      Rui Xu
@Time:        Mar/2026
@Contact:     zaicyxu@gmail.com
@Description: Diverse Prompt-driven Multi-LLM Candidate Generation for Structured Reasoning
"""

import requests
from configuration import Config


class Stage1Generator:
    """
    Generate candidate answers using multiple prompt transformation strategies
    and multiple LLM backends.
    """

    def __init__(self):
        self.base_url = Config.BASE_URL
        self.api_keys = Config.API_KEYS
        self.models = Config.MODELS
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS

    # Base output format constraint
    def _format_instruction(self):
        """
        Enforce a unified structured output format across all prompts.
        """
        return (
            "You must answer strictly using the following structure:\n\n"
            "[Facts]\n"
            "- Key factual statements\n\n"
            "[Reasoning]\n"
            "- Step-by-step reasoning\n\n"
            "[Final Answer]\n"
            "- Concise final answer\n\n"
        )

    # Structured transformation
    def _structured_transform(self, query):
        """
        Transform query into a structured problem decomposition task.
        """
        return (
            "Transform the problem into a structured reasoning process:\n"
            "1. Identify core entities\n"
            "2. Identify relationships\n"
            "3. Decompose into sub-problems\n"
            "4. Solve step by step\n\n"
            f"Problem:\n{query}"
        )

    # Explicit reasoning transformation
    def _reasoning_transform(self, query):
        """
        Force explicit reasoning trajectory.
        """
        return (
            "Solve the problem with explicit reasoning steps:\n"
            "1. State known conditions\n"
            "2. Derive intermediate steps\n"
            "3. Justify each step\n"
            "4. Conclude logically\n\n"
            f"Problem:\n{query}"
        )

    # Example-driven transformation
    def _example_transform(self, query):
        """
        Use analogical reasoning via self-generated example.
        """
        return (
            "First construct a simple analogous example and solve it, "
            "then solve the target problem using similar reasoning.\n\n"
            f"Target Problem:\n{query}"
        )

    # Build all prompt variants
    def _build_prompts(self, query):
        """
        Generate four cognitively distinct prompt variants.
        """
        base = self._format_instruction()

        prompts = {
            "original": base + f"Problem:\n{query}",
            "structured": base + self._structured_transform(query),
            "reasoning": base + self._reasoning_transform(query),
            "example": base + self._example_transform(query),
        }

        return prompts

    # Call LLM API
    def _call_model(self, model_name, prompt):
        """
        Unified API call for all models via compatible endpoint.
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_keys[model_name]}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.models[model_name],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                return f"[ERROR] {response.text}"

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[EXCEPTION] {str(e)}"

    # Public interface
    def generate(self, query):
        """
        Input:
            query (str): natural language question
        Output:
            List[Dict]: candidate answer pool
        """

        prompts = self._build_prompts(query)
        models = ["gpt", "gemini", "qwen", "deepseek"]

        results = []

        for prompt_type, prompt in prompts.items():
            for model in models:
                print(f"[Stage1] {model} | {prompt_type}")
                answer = self._call_model(model, prompt)
                results.append({
                    "model": model,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "answer": answer
                })

        return results