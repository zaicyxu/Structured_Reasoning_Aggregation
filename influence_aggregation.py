#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        influence_aggregation.py
@Author:      Rui Xu
@Contact:     zaicyxu@gmail.com
@Time:        Mar/2026
@Description: Hierarchical Multi-channel Reliability Inference with Interaction-aware Reweighting
"""

import numpy as np
import random


class Stage2Processor:
    def __init__(self, extractor):
        """
        Initialize the processor.
        """
        self.extractor = extractor

    def _extract_units(self, answer_pool):
        """
        Convert raw answers into structured units.

        Each answer is decomposed into fact, reasoning, and result components.
        A unique id is assigned to preserve traceability across channels.
        """
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
        """
        Flatten structured units into three independent sets:
        fact set, reasoning set, and result set.

        Each element retains a reference (uid) to its original unit.
        """
        F, R, Y = [], [], []

        for u in units:
            for f in u["fact"]:
                F.append({"text": f, "uid": u["id"]})

            for r in u["reasoning"]:
                R.append({"text": r, "uid": u["id"]})

            Y.append({"text": u["result"], "uid": u["id"]})

        return F, R, Y

    def _sim(self, a, b):
        """
        Compute similarity between two text segments.
        Uses simple token overlap as a proxy for semantic similarity.
        """
        return len(set(a.split()) & set(b.split()))

    def _agree(self, a, b):
        """
        Compute agreement between two result strings.
        """
        return 1.0 if a == b else -1.0

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

        Energy is defined as the sum of pairwise interactions,
        representing global consistency within the set.
        """
        return M.sum(axis=1)

    def _rejection(self, data, E):
        """
        Perform energy-based rejection sampling.

        Elements are probabilistically accepted based on normalized
        energy scores, approximating a target distribution.
        """
        E = E - np.max(E)
        p = np.exp(E)
        p /= p.sum()

        accept = np.minimum(1.0, p * len(data))

        new_data = []
        idx = []

        for i in range(len(data)):
            if random.random() < accept[i]:
                new_data.append(data[i])
                idx.append(i)

        return new_data, idx

    def _reweight_base(self, M):
        """
        Perform graph-based weight propagation.
        The interaction matrix is normalized row-wise and used
        as a transition operator. Weights are iteratively updated,
        similar to attention or message passing.
        """
        row_sum = np.abs(M).sum(axis=1, keepdims=True) + 1e-8
        A = M / row_sum

        w = np.ones(M.shape[0]) / M.shape[0]

        for _ in range(2):
            w = A.T @ w

        return w

    def _cross_influence(self, src_set, src_w, tgt_set):
        """
        Compute cross-channel influence.
        Influence is propagated based on shared unit ids (uid),
        allowing weak coupling between channels.
        """
        influence = np.zeros(len(tgt_set))

        for i, tgt in enumerate(tgt_set):
            for j, src in enumerate(src_set):
                if tgt["uid"] == src["uid"]:
                    influence[i] += src_w[j]

        return influence

    def _softmax(self, x):
        """
        Apply softmax normalization.
        Converts arbitrary scores into a probability distribution.
        """
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def _process_channel(self, dataset, mode):
        """
        Full pipeline for a single channel.
        """
        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        E = self._energy(M)

        dataset, idx = self._rejection(dataset, E)

        if len(dataset) == 0:
            return [], np.array([])

        M = self._build_M(dataset, mode)
        w = self._reweight_base(M)

        return dataset, w

    def _aggregate(self, Y, w):
        """
        Aggregate result set into pseudo ground truth.
        Performs weighted voting over candidate results.
        """
        score = {}
        for item, weight in zip(Y, w):
            key = item["text"]
            score[key] = score.get(key, 0) + weight

        best = max(score.items(), key=lambda x: x[1])[0]
        return best, score

    def _confidence(self, wf, wr, wy):
        """
        Compute overall confidence.
        """
        def ent(w):
            if len(w) == 0:
                return 1
            return -np.sum(w * np.log(w + 1e-8)) / np.log(len(w))

        return float((1 - ent(wf) + 1 - ent(wr) + 1 - ent(wy)) / 3)

    def process(self, answer_pool):
        """
        Full Stage2 pipeline.
        """
        units = self._extract_units(answer_pool)
        F, R, Y = self._build_sets(units)

        # independent inference
        F, wf = self._process_channel(F, "sim")
        R, wr = self._process_channel(R, "sim")
        Y, wy = self._process_channel(Y, "agree")

        if len(Y) == 0:
            return {}

        # weak coupling coefficients
        lambda_f = 0.3
        lambda_r = 0.3
        lambda_y = 0.5

        # cross-channel influence
        inf_F = self._cross_influence(Y, wy, F)
        inf_R = self._cross_influence(F, wf, R)
        inf_Y = self._cross_influence(F, wf, Y) + self._cross_influence(R, wr, Y)

        # adjust weights
        wf = self._softmax(wf + lambda_f * inf_F)
        wr = self._softmax(wr + lambda_r * inf_R)
        wy = self._softmax(wy + lambda_y * inf_Y)

        # final aggregation
        result, dist = self._aggregate(Y, wy)
        conf = self._confidence(wf, wr, wy)

        return {
            "fact_weights": wf.tolist(),
            "reasoning_weights": wr.tolist(),
            "result_weights": wy.tolist(),
            "pseudo_groundtruth": result,
            "confidence": conf
        }