#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        evaluator.py
@Author:      Rui Xu
@Time:        Mar/2026
@Version:     0.1.0
@Description: Automated Batch Evaluation and CSV Signal Export
"""

import json
import pandas as pd
from main import Pipeline


class PipelineEvaluator:
    def __init__(self):
        """
        Initialize the evaluator with self-contained pipeline.
        """
        self.pipeline = Pipeline()
        self.results_cache = []

    def run_inference(self, q_id, query):
        """
        Run inference and capture Stage 2/3 signals for CSV storage.
        """
        try:
            output = self.pipeline.run(query, iterations=1)
            if not output: return

            s2_conf = output.get("stage2", {}).get("confidence", 0)
            s3_ranking = output.get("stage3", {}).get("ranking", [])

            if s3_ranking:
                top_ans = s3_ranking[0]
                sig = top_ans.get("signal", {})

                self.results_cache.append({
                    "ID": q_id,
                    "Confidence": s2_conf,
                    "Validity (V)": sig.get("validity", 0),
                    "Completeness (C)": sig.get("completeness", 0),
                    "Consistency (C)": sig.get("consistency", 0),
                    "Utility (U)": sig.get("utility", 0)
                })
        except Exception:
            pass

    def save_to_csv(self, output_file):
        """
        Export results to CSV with a summary average row.
        """
        if not self.results_cache: return

        df = pd.DataFrame(self.results_cache)
        avg_values = df.mean(numeric_only=True).to_dict()
        avg_values["ID"] = "AVG"

        df_final = pd.concat([df, pd.DataFrame([avg_values])], ignore_index=True)
        columns = ["ID", "Confidence", "Validity (V)", "Completeness (C)", "Consistency (C)", "Utility (U)"]
        df_final[columns].to_csv(output_file, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    # Config at entry point
    input_json = "questions.json"
    output_csv = "evaluation_results.csv"

    # Data loading at entry point
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_cases = data if isinstance(data, list) else data.get("questions", [])
    except Exception:
        test_cases = []

    # Execution loop
    evaluator = PipelineEvaluator()
    for case in test_cases:
        evaluator.run_inference(case.get("id", "N/A"), case.get("question", ""))

    # Final export
    evaluator.save_to_csv(output_csv)