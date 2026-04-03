#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        influence_aggregation.py
@Author:      Rui Xu
@Time:        Mar/2026
@Version:     0.5.2
@Description: Adaptive Hierarchical Multi-channel Inference with Feedback Control
"""

import numpy as np
import random


class Stage2Processor:
    def __init__(self, extractor):
        self.extractor = extractor

    # Control Parsing
    def _parse_control(self, control_signal):
        """
        Extract Stage2 control parameters.
        """
        if not control_signal or "stage2" not in control_signal:
            return 0.5, 0.5, 0.5  # default

        ctrl = control_signal["stage2"]

        return (
            ctrl.get("rejection_strength", 0.5),
            ctrl.get("reweight_strength", 0.5),
            ctrl.get("coupling_strength", 0.5),
        )

    # Structure Extraction
    def _extract_units(self, answer_pool):
        units = []
        for idx, item in enumerate(answer_pool):
            s = self.extractor(item["answer"])
            units.append({
                "id": idx,
                "fact": s.get("fact", []),
                "reasoning": s.get("reasoning", []),
                "result": s.get("result", "")
            })
        return units

    def _build_sets(self, units):
        F, R, Y = [], [], []
        for u in units:
            for f in u["fact"]:
                F.append({"text": f, "uid": u["id"]})

            for r in u["reasoning"]:
                R.append({"text": r, "uid": u["id"]})

            Y.append({"text": u["result"], "uid": u["id"]})

        return F, R, Y

    # Interaction
    def _sim(self, a, b):
        return len(set(a.split()) & set(b.split()))

    def _agree(self, a, b):
        return 1.0 if a == b else -1.0

    def _build_M(self, dataset, mode):
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
        return M.sum(axis=1)

    # Controlled Rejection
    def _rejection(self, data, E, rejection_strength):
        """
        Adaptive rejection sampling.
        rejection_strength controls sharpness of acceptance distribution.
        """
        E = E - np.max(E)
        p = np.exp(E)
        p /= p.sum()

        gamma = 1 + rejection_strength * 2  # sharpening factor
        p = p ** gamma
        p /= p.sum()

        accept = np.minimum(1.0, p * len(data))
        new_data = []
        idx = []

        for i in range(len(data)):
            if random.random() < accept[i]:
                new_data.append(data[i])
                idx.append(i)

        return new_data, idx

    # Adaptive Reweight
    def _reweight_base(self, M, reweight_strength):
        """
        Graph propagation with adaptive depth.
        """
        row_sum = np.abs(M).sum(axis=1, keepdims=True) + 1e-8
        A = M / row_sum
        w = np.ones(M.shape[0]) / M.shape[0]
        steps = int(1 + reweight_strength * 4)

        for _ in range(steps):
            w = A.T @ w

        return w

    # Cross Influence
    def _cross_influence(self, src_set, src_w, tgt_set):
        influence = np.zeros(len(tgt_set))

        for i, tgt in enumerate(tgt_set):
            for j, src in enumerate(src_set):
                if tgt["uid"] == src["uid"]:
                    influence[i] += src_w[j]

        return influence

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    # Channel Processing
    def _process_channel(self, dataset, mode, r_strength, rw_strength):
        """
        Full controlled pipeline for one channel.
        """
        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        E = self._energy(M)

        dataset, _ = self._rejection(dataset, E, r_strength)

        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        w = self._reweight_base(M, rw_strength)

        return dataset, w

    # Aggregation
    def _aggregate(self, Y, w):
        score = {}
        for item, weight in zip(Y, w):
            key = item["text"]
            score[key] = score.get(key, 0) + weight

        best = max(score.items(), key=lambda x: x[1])[0]
        return best, score

    def _confidence(self, wf, wr, wy):
        def ent(w):
            if len(w) == 0:
                return 1
            return -np.sum(w * np.log(w + 1e-8)) / np.log(len(w))

        return float((1 - ent(wf) + 1 - ent(wr) + 1 - ent(wy)) / 3)

    # Main Pipeline
    def process(self, answer_pool, control_signal=None):
        """
        Stage2 with adaptive control input.
        """
        # parse control
        r_strength, rw_strength, c_strength = self._parse_control(control_signal)

        # extract
        units = self._extract_units(answer_pool)
        F, R, Y = self._build_sets(units)

        # independent inference
        F, wf = self._process_channel(F, "sim", r_strength, rw_strength)
        R, wr = self._process_channel(R, "sim", r_strength, rw_strength)
        Y, wy = self._process_channel(Y, "agree", r_strength, rw_strength)

        if len(Y) == 0:
            return {}

        # dynamic coupling
        lambda_f = 0.2 + 0.5 * c_strength
        lambda_r = 0.2 + 0.5 * c_strength
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