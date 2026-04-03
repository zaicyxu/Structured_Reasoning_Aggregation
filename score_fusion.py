#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        score_fusion.py
@Author:      Rui Xu
@Contact:     zaicyxu@gmail.com
@Time:        Mar/2026
@Version:     0.3.2
@Description: Evaluation-driven Signal Projection and Reliability-Quality Fusion with Full Signal Preservation
"""

import numpy as np
import json


class Stage3Processor:
    def __init__(self, llm_call):
        """
        Initialize Stage3 processor.
        """
        self.llm_call = llm_call

    def _build_prompt(self, answer):
        """
        Construct the evaluation prompt.
        """
        prompt = f"""
                You are an expert evaluator.

                Evaluate the following answer based on FOUR criteria:

                1. Validity (0-5)
                - 5: Fully factually correct
                - 3: Mostly correct with minor errors
                - 1: Major factual issues
                - 0: Completely incorrect

                2. Completeness (0-5)
                - 5: Fully addresses the question
                - 3: Partially complete
                - 1: Missing key parts
                - 0: Severely incomplete

                3. Consistency (0-5)
                - 5: Fully logically consistent
                - 3: Minor inconsistencies
                - 1: Major contradictions
                - 0: Completely inconsistent

                4. Utility (0-5)
                - 5: Highly useful and actionable
                - 3: Moderately useful
                - 1: Limited usefulness
                - 0: Not useful

                Answer:
                \"\"\"{answer}\"\"\"

                Output ONLY JSON in the format:
                {{
                  "validity": int,
                  "completeness": int,
                  "consistency": int,
                  "utility": int
                }}
                """
        return prompt

    def _evaluate(self, candidates):
        """
        Evaluate all candidate answers using the LLM.
        """
        Z = []

        for ans in candidates:
            prompt = self._build_prompt(ans)
            response = self.llm_call(prompt)

            try:
                z = json.loads(response)
            except:
                # Fallback ensures robustness if LLM output is malformed
                z = {
                    "validity": 0,
                    "completeness": 0,
                    "consistency": 0,
                    "utility": 0
                }

            Z.append(z)

        return Z

    def _softmax(self, w):
        """
        Normalize raw weights into a probability distribution.
        """
        w = np.array(w)
        w = w - np.max(w)
        e = np.exp(w)
        return e / np.sum(e)

    def _project(self, z):
        """
        Project multi-dimensional evaluation signal into scalar quality score.
        """
        vec = np.array([
            z["validity"],
            z["completeness"],
            z["consistency"],
            z["utility"]
        ]) / 5.0

        # uncertainty estimation
        uncertainty = np.var(vec)

        # weighted aggregation
        W = np.array([0.3, 0.25, 0.25, 0.2])
        base_score = np.dot(W, vec)

        # non-linear transformation
        score = np.tanh(base_score)

        # penalize uncertainty
        score = score * (1 - uncertainty)

        return score, uncertainty

    def _aggregate(self, W, Z):
        """
        Fuse reliability weights and quality signals.

        This implements multiplicative fusion:
        score = w^alpha * q^beta
        """
        scores = []
        qualities = []
        uncertainties = []

        alpha = 0.6
        beta = 0.8

        for w, z in zip(W, Z):
            q, u = self._project(z)

            score = (w ** alpha) * (q ** beta)

            scores.append(score)
            qualities.append(q)
            uncertainties.append(u)

        return np.array(scores), qualities, uncertainties

    def _rank(self, candidates, scores, W, Z, Q, U, top_k=3):
        """
        Rank candidates and preserve full signal information.
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

    def process(self, candidates, weights):
        """
        Execute full Stage3 pipeline.
        """
        if len(candidates) == 0:
            return {}

        W = self._softmax(weights)
        Z = self._evaluate(candidates)

        scores, Q, U = self._aggregate(W, Z)

        topk, full = self._rank(candidates, scores, W, Z, Q, U)

        return {
            "ranking": topk,
            "full": full,
            "global": {
                "weights": W.tolist(),
                "signals": Z
            }
        }