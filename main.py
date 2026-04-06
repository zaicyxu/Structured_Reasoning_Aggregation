#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        main.py
@Author:      Rui Xu
@Time:        Mar/2026
@Version:     0.2.3
@Description: Full Pipeline Orchestration
"""

from rea_fuser import Stage1Generator
from influence_aggregation import Stage2Processor
from score_fusion import Stage3Processor
from feedback_controller import Stage4Controller


class Pipeline:
    """
    Main pipeline controller.

    Responsibilities:
    - Coordinate Stage1 → Stage4
    - Pass signals correctly
    - Maintain clean data flow
    """

    def __init__(self, llm_call):
        """
        Initialize all stages.
        """
        self.stage1 = Stage1Generator()
        self.stage2 = Stage2Processor(llm_call)
        self.stage3 = Stage3Processor(llm_call)
        self.stage4 = Stage4Controller()

    def run(self, query, iterations=1):
        """
        Execute the full pipeline.
        """
        control_signal = None  # cold start
        for step in range(iterations):

            print(f"\n Iteration {step+1}")

            # Stage1: Candidate Generation
            print("\n[Stage1] Generating candidates...")
            answer_pool = self.stage1.generate(query, control_signal)

            # extract candidate answers
            candidates = [item["answer"] for item in answer_pool]

            # Stage2: Structured Inference
            print("[Stage2] Running structured inference...")
            stage2_output = self.stage2.process(answer_pool, control_signal)

            if not stage2_output:
                print("Stage2 returned empty result.")
                return {}

            # weights for Stage3
            weights = stage2_output["result_weights"]

            # Stage3: Evaluation + Ranking
            print("[Stage3] Evaluating and ranking...")
            stage3_output = self.stage3.process(candidates, weights)

            if not stage3_output:
                print("Stage3 returned empty result.")
                return {}

            # Stage4: Feedback Control
            print("[Stage4] Generating feedback signal...")
            control_signal = self.stage4.process(stage2_output, stage3_output)

            print("\n[Feedback]")
            print(control_signal)

        # Final Output
        final_results = {
            "query": query,
            "stage2": stage2_output,
            "stage3": stage3_output,
            "control": control_signal
        }

        return final_results


# ENTRY POINT
if __name__ == "__main__":
    def llm_call(prompt):
        """
        Unified LLM interface.
        """
        raise NotImplementedError("Please implement llm_call function.")

    pipeline = Pipeline(llm_call)
    query = input("Enter your question:\n> ")
    results = pipeline.run(query, iterations=1)
    print("\n FINAL RESULT:")

    if results:
        ranking = results["stage3"]["ranking"]
        for i, item in enumerate(ranking):
            print(f"\nTop {i+1}:")
            print("Answer:", item["answer"])
            print("Score:", item["score"])
            print("Weight:", item["weight"])
            print("Quality:", item["quality"])
            print("Uncertainty:", item["uncertainty"])