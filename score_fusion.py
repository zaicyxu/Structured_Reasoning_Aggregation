#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        score_fusion.py
@Author:      Rui Xu
@Contact:     zaicyxu@gmail.com
@Time:        Mar/2026
@Description: Evaluation-driven Signal Projection and Reliability-Quality Fusion for Final Decision Making
"""

import numpy as np
import json


class Stage3Processor:
    def __init__(self, llm_call):
        self.llm_call = llm_call

    # Prompt Engineering
    def _build_prompt(self, answer):
        """
        Construct evaluation prompt with strict scoring rubric.
        The LLM is required to output structured JSON.
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

    # LLM Evaluation
    def _evaluate(self, candidates):
        """
        Call LLM to generate evaluation signals Z.
        """
        Z = []

        for ans in candidates:
            prompt = self._build_prompt(ans)
            response = self.llm_call(prompt)
            try:
                z = json.loads(response)
            except:
                # fallback if parsing fails
                z = {
                    "validity": 0,
                    "completeness": 0,
                    "consistency": 0,
                    "utility": 0
                }

            Z.append(z)
        return Z

    # Weight Normalization
    def _softmax(self, w):
        w = np.array(w)
        w = w - np.max(w)
        e = np.exp(w)
        return e / np.sum(e)

    # Signal Projection
    def _project(self, z):
        """
        Project multi-dimensional signal into scalar quality score.
        """
        vec = np.array([
            z["validity"],
            z["completeness"],
            z["consistency"],
            z["utility"]
        ]) / 5.0

        # uncertainty = variance
        uncertainty = np.var(vec)

        # weighted linear projection
        W = np.array([0.3, 0.25, 0.25, 0.2])
        base_score = np.dot(W, vec)

        # non-linear transformation
        score = np.tanh(base_score)

        # uncertainty penalty
        score = score * (1 - uncertainty)

        return score

    # Aggregation
    def _aggregate(self, W, Z):
        """
        Fuse reliability (W) and quality (Z).
        Uses multiplicative fusion with exponents.
        """
        scores = []

        for w, z in zip(W, Z):
            q = self._project(z)

            # fusion (tunable)
            alpha = 0.6  # reliability importance
            beta = 0.8   # quality importance

            score = (w ** alpha) * (q ** beta)
            scores.append(score)

        return np.array(scores)


    # Ranking
    def _rank(self, candidates, scores, top_k=3):
        idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in idx:
            results.append({
                "answer": candidates[i],
                "score": float(scores[i])
            })

        return results

    # Main Pipeline
    def process(self, candidates, weights):
        """
            Top-K ranked answers
        """
        if len(candidates) == 0:
            return []

        # normalize weights
        W = self._softmax(weights)

        # LLM evaluation
        Z = self._evaluate(candidates)

        # aggregation
        scores = self._aggregate(W, Z)

        # ranking
        results = self._rank(candidates, scores, top_k=3)

        return results