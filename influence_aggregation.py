#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        influence_aggregation.py
@Author:      Rui Xu
@Time:        Mar/2026
@Version:     0.6.1
@Description: Fully Integrated Hierarchical Multi-channel Inference with LLM-based Structuring and Feedback Control
"""

import numpy as np
import random
import json


class Stage2Processor:
    """
    Stage2 Processor:
    - LLM-based structured decomposition
    - Hierarchical (Fact / Reasoning / Result) inference
    - Controlled rejection sampling
    - Graph-based reweighting
    - Cross-channel weak coupling
    """

    def __init__(self, llm_call):
        """
        Initialize with LLM interface.
        """
        self.llm_call = llm_call

    # Control Parsing (Stage4)
    def _parse_control(self, control_signal):
        """
        Extract adaptive parameters from Stage4.
        """
        if not control_signal or "stage2" not in control_signal:
            return 0.5, 0.5, 0.5

        ctrl = control_signal["stage2"]

        return (
            ctrl.get("rejection_strength", 0.5),
            ctrl.get("reweight_strength", 0.5),
            ctrl.get("coupling_strength", 0.5),
        )

    # LLM structured extraction
    def _build_extraction_prompt(self, answer):
        """
        Build strict prompt for structured decomposition.

        Forces LLM to output JSON with:
        fact / reasoning / result
        """
        return f"""
        You are a structured reasoning parser.

        Decompose the following answer into THREE components:

        1. Facts:
        - Objective factual statements only

        2. Reasoning:
        - Logical steps and inference process

        3. Result:
        - Final concise conclusion

        Answer:
        \"\"\"{answer}\"\"\"

        Output ONLY JSON:
        {{
            "fact": [str],
            "reasoning": [str],
            "result": str
        }}
        """

    def _extract_units(self, answer_pool):
        """
        Perform LLM-based structured extraction.
        """
        units = []

        for idx, item in enumerate(answer_pool):
            prompt = self._build_extraction_prompt(item["answer"])
            response = self.llm_call(prompt)
            try:
                parsed = json.loads(response)
            except:
                parsed = {"fact": [], "reasoning": [], "result": ""}

            units.append({
                "id": idx,
                "fact": parsed.get("fact", []),
                "reasoning": parsed.get("reasoning", []),
                "result": parsed.get("result", "")
            })

        return units

    # Build channel set
    def _build_sets(self, units):
        """
        Flatten units into three independent sets.
        """
        F, R, Y = [], [], []

        for u in units:
            for f in u["fact"]:
                F.append({"text": f, "uid": u["id"]})

            for r in u["reasoning"]:
                R.append({"text": r, "uid": u["id"]})
            Y.append({"text": u["result"], "uid": u["id"]})

        return F, R, Y

    # Similarity / Agreement
    def _sim(self, a, b):
        """
        Compute normalized similarity (Jaccard).
        """
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / (len(sa | sb) + 1e-8)

    def _agree(self, a, b):
        """
        Binary agreement for result comparison.
        """
        return 1.0 if a == b else -1.0

    # Interaction Martix
    def _build_M(self, dataset, mode):
        """
        Construct pairwise interaction matrix.
        """
        n = len(dataset)
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                if mode == "sim":
                    M[i, j] = self._sim(dataset[i]["text"], dataset[j]["text"])
                else:
                    M[i, j] = self._agree(dataset[i]["text"], dataset[j]["text"])

        return M

    def _energy(self, M):
        """
        Compute energy score for each element.
        Represents global consistency.
        """
        return M.sum(axis=1)

    # Rejection sampling
    def _rejection(self, data, E, rejection_strength):
        """
        Adaptive rejection sampling.
        """
        E = E - np.max(E)
        p = np.exp(E)
        p /= p.sum()

        # sharpen distribution
        gamma = 1 + rejection_strength * 3
        p = p ** gamma
        p /= p.sum()

        accept_prob = np.minimum(1.0, p * len(data))
        new_data = []

        for i in range(len(data)):
            if random.random() < accept_prob[i]:
                new_data.append(data[i])

        return new_data

    # Graph reweighting
    def _reweight(self, M, reweight_strength):
        """
        Graph-based message passing.
        Depth controlled by reweight_strength.
        """
        row_sum = np.abs(M).sum(axis=1, keepdims=True) + 1e-8
        A = M / row_sum

        w = np.ones(M.shape[0]) / M.shape[0]

        steps = int(1 + reweight_strength * 5)

        for _ in range(steps):
            w = A.T @ w

        return w

    # Cross-channel coupline
    def _cross_influence(self, src_set, src_w, tgt_set):
        """
        Propagate influence across channels via shared uid.
        """
        influence = np.zeros(len(tgt_set))

        for i, tgt in enumerate(tgt_set):
            for j, src in enumerate(src_set):
                if tgt["uid"] == src["uid"]:
                    influence[i] += src_w[j]

        return influence

    def _softmax(self, x):
        """
        Standard softmax normalization.
        """
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # Signal channel pipeline
    def _process_channel(self, dataset, mode, r_strength, rw_strength):
        """
        Full pipeline for one channel.
        """
        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        E = self._energy(M)

        dataset = self._rejection(dataset, E, r_strength)

        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        w = self._reweight(M, rw_strength)

        return dataset, w

    # Aggregation
    def _aggregate(self, Y, w):
        """
        Aggregate result channel into pseudo ground truth.
        """
        score = {}

        for item, weight in zip(Y, w):
            key = item["text"]
            score[key] = score.get(key, 0) + weight

        best = max(score.items(), key=lambda x: x[1])[0]
        return best, score

    # Confidence
    def _confidence(self, wf, wr, wy):
        """
        Compute confidence using entropy.
        """
        def entropy(w):
            if len(w) == 0:
                return 1
            return -np.sum(w * np.log(w + 1e-8)) / np.log(len(w))

        return float((1 - entropy(wf) + 1 - entropy(wr) + 1 - entropy(wy)) / 3)

    # Main pipeline
    def process(self, answer_pool, control_signal=None):
        """
        Full Stage2 pipeline.
        """
        # parse control
        r_strength, rw_strength, c_strength = self._parse_control(control_signal)

        # extraction
        units = self._extract_units(answer_pool)

        # build sets
        F, R, Y = self._build_sets(units)

        # independent inference
        F, wf = self._process_channel(F, "sim", r_strength, rw_strength)
        R, wr = self._process_channel(R, "sim", r_strength, rw_strength)
        Y, wy = self._process_channel(Y, "agree", r_strength, rw_strength)

        if len(Y) == 0:
            return {}

        # adaptive coupling
        lambda_f = 0.2 + 0.6 * c_strength
        lambda_r = 0.2 + 0.6 * c_strength
        lambda_y = 0.3 + 0.7 * c_strength

        inf_F = self._cross_influence(Y, wy, F)
        inf_R = self._cross_influence(F, wf, R)
        inf_Y = self._cross_influence(F, wf, Y) + self._cross_influence(R, wr, Y)

        wf = self._softmax(wf + lambda_f * inf_F)
        wr = self._softmax(wr + lambda_r * inf_R)
        wy = self._softmax(wy + lambda_y * inf_Y)

        # aggregation
        result, dist = self._aggregate(Y, wy)
        conf = self._confidence(wf, wr, wy)

        return {
            "fact_weights": wf.tolist(),
            "reasoning_weights": wr.tolist(),
            "result_weights": wy.tolist(),
            "pseudo_groundtruth": result,
            "confidence": conf
        }