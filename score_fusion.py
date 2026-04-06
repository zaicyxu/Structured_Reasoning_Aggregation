#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        score_fusion.py
@Author:      Rui Xu
@Time:        Mar/2026
@Version:     0.4.2
@Description: Stable Evaluation-driven Signal Projection and Reliability-Quality Fusion
"""

import numpy as np
import json


class Stage3Processor:
    """
    Stage3:
    - LLM-based evaluation (Z)
    - Signal projection (quality)
    - Reliability-quality fusion
    - Full signal preservation
    """

    def __init__(self, llm_call):
        self.llm_call = llm_call

    # Prompt
    def _build_prompt(self, answer):
        """
        Construct strict evaluation prompt.
        """
        return f"""
        You are an expert evaluator.

        Evaluate the following answer based on FOUR criteria:

        1. Validity (0-5)
        2. Completeness (0-5)
        3. Consistency (0-5)
        4. Utility (0-5)

        Answer:
        \"\"\"{answer}\"\"\"

        Output ONLY JSON:
        {{
          "validity": int,
          "completeness": int,
          "consistency": int,
          "utility": int
        }}
        """


    # LLM evaluation
    def _evaluate(self, candidates):
        """
        Generate evaluation signals Z using LLM.
        Includes safety parsing and clamping.
        """
        Z = []

        for ans in candidates:
            prompt = self._build_prompt(ans)
            response = self.llm_call(prompt)

            try:
                z = json.loads(response)
            except:
                z = {}

            # safe parsing + clamp
            z_clean = {
                "validity": int(np.clip(z.get("validity", 0), 0, 5)),
                "completeness": int(np.clip(z.get("completeness", 0), 0, 5)),
                "consistency": int(np.clip(z.get("consistency", 0), 0, 5)),
                "utility": int(np.clip(z.get("utility", 0), 0, 5)),
            }

            Z.append(z_clean)

        return Z

    # Softmax
    def _softmax(self, w):
        """
        Normalize weights into probability distribution.
        """
        w = np.array(w)
        w = w - np.max(w)
        e = np.exp(w)
        return e / (np.sum(e) + 1e-8)

    # Signal projection
    def _project(self, z):
        """
        Convert multi-dimensional signal into scalar quality.
        """
        vec = np.array([
            z["validity"],
            z["completeness"],
            z["consistency"],
            z["utility"]
        ]) / 5.0

        # uncertainty
        uncertainty = np.var(vec)

        # weighted sum
        W = np.array([0.3, 0.25, 0.25, 0.2])
        base = np.dot(W, vec)

        # sigmoid projection (better than tanh here)
        q = 1 / (1 + np.exp(-4 * (base - 0.5)))

        # uncertainty penalty
        q = q * (1 - uncertainty)

        # clamp
        q = float(np.clip(q, 1e-6, 1.0))
        return q, float(uncertainty)

    # Fusion
    def _aggregate(self, W, Z):
        """
        Reliability-quality fusion in log-space.
        score = exp(alpha log w + beta log q)
        """
        scores = []
        qualities = []
        uncertainties = []

        alpha = 0.6
        beta = 0.8

        for w, z in zip(W, Z):
            q, u = self._project(z)

            # log-space fusion (numerically stable)
            log_score = alpha * np.log(w + 1e-8) + beta * np.log(q + 1e-8)
            score = np.exp(log_score)

            scores.append(score)
            qualities.append(q)
            uncertainties.append(u)

        return np.array(scores), qualities, uncertainties

    # Ranking
    def _rank(self, candidates, scores, W, Z, Q, U, top_k=3):
        """
        Rank answers and preserve full signal trace.
        """
        idx = np.argsort(scores)[::-1]

        full = []
        for i in idx:
            full.append({
                "answer": candidates[i],
                "score": float(scores[i]),
                "weight": float(W[i]),
                "signal": Z[i],
                "quality": float(Q[i]),
                "uncertainty": float(U[i])
            })

        return full[:top_k], full

    # Main pipeline
    def process(self, candidates, weights):
        """
        Execute Stage3 pipeline.
        """
        if len(candidates) == 0:
            return {}

        # normalize reliability
        W = self._softmax(weights)

        # LLM evaluation
        Z = self._evaluate(candidates)

        # fusion
        scores, Q, U = self._aggregate(W, Z)

        # ranking
        topk, full = self._rank(candidates, scores, W, Z, Q, U)

        return {
            "ranking": topk,
            "full": full,
            "global": {
                "weights": W.tolist(),
                "signals": Z,
                "qualities": Q,
                "uncertainties": U
            }
        }