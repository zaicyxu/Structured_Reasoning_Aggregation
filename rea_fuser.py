#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        rea_fuser.py
@Author:      Rui Xu
@Time:        Mar/2026
@Contact:     zaicyxu@gmail.com
@Version:     0.3.0
@Description: Diverse Prompt-driven Multi-LLM Candidate Generation with Adaptive Control
"""

import requests
import random
from configuration import Config


class Stage1Generator:
    """
    Generate candidate answers using multiple prompt strategies
    with adaptive control from Stage4.
    """

    def __init__(self):
        self.base_url = Config.BASE_URL
        self.api_keys = Config.API_KEYS
        self.models = Config.MODELS

        self.default_temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS

    # Format Constraint
    def _format_instruction(self):
        """
        Enforce a unified structured output format.
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

    # Prompt Transformations
    def _structured_transform(self, query):
        """
        Enforce graph-structured problem decomposition.
        """
        return (
            "You must solve the problem using a STRICT structured decomposition.\n\n"

            "Follow EXACTLY these steps:\n\n"

            "[Step 1: Entity Extraction]\n"
            "- Identify all key entities, variables, and concepts\n"
            "- Do NOT skip implicit elements\n\n"

            "[Step 2: Relation Mapping]\n"
            "- Explicitly describe relationships between entities\n"
            "- Use clear relational statements (e.g., causes, depends on, influences)\n\n"

            "[Step 3: Sub-problem Decomposition]\n"
            "- Break the problem into smaller solvable units\n"
            "- Each sub-problem must be independently meaningful\n\n"

            "[Step 4: Dependency Graph Reasoning]\n"
            "- Solve sub-problems in dependency order\n"
            "- Each step must reference previous results\n\n"

            "[Step 5: Final Synthesis]\n"
            "- Integrate all sub-results into the final answer\n\n"

            "Constraints:\n"
            "- Do NOT jump directly to conclusions\n"
            "- Do NOT skip any step\n"
            "- Every reasoning step must map to previous structure\n\n"

            f"Problem:\n{query}"
        )

    def _reasoning_transform(self, query):
        """
        Enforce strict causal and verifiable reasoning chain.
        """
        return (
            "You must solve the problem using a STRICT step-by-step logical reasoning process.\n\n"

            "Follow EXACTLY this reasoning protocol:\n\n"

            "[Step 1: Known Information]\n"
            "- List all given facts explicitly\n"
            "- Do NOT assume unstated facts\n\n"

            "[Step 2: Goal Definition]\n"
            "- Clearly define what needs to be solved\n\n"

            "[Step 3: Step-wise Deduction]\n"
            "- Each step must follow logically from previous steps\n"
            "- Each step must include justification\n"
            "- Avoid large jumps in reasoning\n\n"

            "[Step 4: Intermediate Verification]\n"
            "- After key steps, verify correctness\n"
            "- Check for contradictions or inconsistencies\n\n"

            "[Step 5: Final Conclusion]\n"
            "- Provide final answer strictly based on derived steps\n\n"

            "Constraints:\n"
            "- NO intuitive leaps\n"
            "- NO missing steps\n"
            "- Each step must be logically grounded\n\n"

            f"Problem:\n{query}"
        )

    def _example_transform(self, query):
        """
        Enforce analogical reasoning with explicit mapping.
        """
        return (
            "You must solve the problem using ANALOGICAL reasoning.\n\n"

            "Follow EXACTLY this process:\n\n"

            "[Step 1: Construct an Analogue Problem]\n"
            "- Create a simpler but structurally similar problem\n"
            "- The analogy must preserve the core logic\n\n"

            "[Step 2: Solve the Analogue]\n"
            "- Fully solve the simpler problem\n"
            "- Show all reasoning steps\n\n"

            "[Step 3: Mapping]\n"
            "- Explicitly map elements from analogue → original problem\n"
            "- Explain correspondence clearly\n\n"

            "[Step 4: Transfer Reasoning]\n"
            "- Apply the same reasoning pattern to the original problem\n\n"

            "[Step 5: Final Answer]\n"
            "- Provide final answer for the original problem\n\n"

            "Constraints:\n"
            "- Analogy must NOT be trivial\n"
            "- Mapping must be explicit\n"
            "- Do NOT skip the transfer step\n\n"

            f"Target Problem:\n{query}"
        )

    # Prompt Builder
    def _build_prompts(self, query):
        """
        Build all candidate prompt variants.
        """
        base = self._format_instruction()
        return {
            "original": base + f"Problem:\n{query}",
            "structured": base + self._structured_transform(query),
            "reasoning": base + self._reasoning_transform(query),
            "example": base + self._example_transform(query),
        }

    # Control Parsing
    def _parse_control(self, control_signal):
        """
        Extract Stage1 control parameters.
        """
        if not control_signal or "stage1" not in control_signal:
            return self.default_temperature, {
                "original": 0.25,
                "structured": 0.25,
                "reasoning": 0.25,
                "example": 0.25
            }

        stage1_ctrl = control_signal["stage1"]
        temperature = stage1_ctrl.get("temperature", self.default_temperature)
        prompt_weights = stage1_ctrl.get("prompt_weights", {})

        return temperature, prompt_weights

    # Prompt Sampling
    def _sample_prompts(self, prompts, weights, total_samples=4):
        """
        Sample prompt types based on weights.
        This replaces uniform enumeration with adaptive sampling.
        """
        types = list(prompts.keys())

        if not weights:
            return types
        probs = [weights.get(t, 0.0) for t in types]

        # normalize
        s = sum(probs)
        if s == 0:
            probs = [1.0 / len(types)] * len(types)
        else:
            probs = [p / s for p in probs]

        sampled = random.choices(types, weights=probs, k=total_samples)

        return list(set(sampled))

    # LLM Call
    def _call_model(self, model_name, prompt, temperature):
        """
        Unified API call with dynamic temperature.
        """
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_keys[model_name]}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.models[model_name],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

            if response.status_code != 200:
                return f"[ERROR] {response.text}"

            return response.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[EXCEPTION] {str(e)}"

    # Public Interface
    def generate(self, query, control_signal=None):
        """
        Generate candidate answers with adaptive control.
        """

        temperature, prompt_weights = self._parse_control(control_signal)

        prompts = self._build_prompts(query)

        selected_types = self._sample_prompts(prompts, prompt_weights)
        models = ["gpt", "gemini", "qwen", "deepseek"]
        results = []

        for prompt_type in selected_types:
            prompt = prompts[prompt_type]

            for model in models:
                print(f"[Stage1] {model} | {prompt_type} | temp={temperature:.2f}")

                answer = self._call_model(model, prompt, temperature)

                results.append({
                    "model": model,
                    "prompt_type": prompt_type,
                    "temperature": temperature,
                    "prompt": prompt,
                    "answer": answer
                })

        return results