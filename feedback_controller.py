#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project:     Structured Reasoning Aggregation
@File:        feedback_controller.py
@Author:      Rui Xu
@Contact:     zaicyxu@gmail.com
@Time:        Mar/2026
@Version:     0.2.8
@Description: Adaptive feedback controller for Stage1 and Stage2 (main-driven architecture)
"""

import numpy as np


class Stage4Controller:
    def __init__(self):
        """
        Initialize controller with predefined bounds and defaults.
        """
        self.min_temp = 0.2
        self.max_temp = 1.2

    # Utility Functions
    def _entropy(self, w):
        """
        Compute normalized entropy of weight distribution.

        Measures how dispersed the weight distribution is.
        High entropy = high uncertainty.
        """
        w = np.array(w)
        if len(w) == 0:
            return 1.0

        w = w / (np.sum(w) + 1e-8)
        return -np.sum(w * np.log(w + 1e-8)) / np.log(len(w))

    def _variance_signal(self, Z):
        """
        Compute average variance across evaluation signals.
        Captures inconsistency among validity/completeness/etc.
        """
        if not Z:
            return 1.0

        vars_ = []
        for z in Z:
            vec = np.array([
                z.get("validity", 0),
                z.get("completeness", 0),
                z.get("consistency", 0),
                z.get("utility", 0)
            ]) / 5.0

            vars_.append(np.var(vec))

        return float(np.mean(vars_))

    # Stage1 Control
    def _control_stage1(self, confidence, entropy, variance):
        """
        Generate executable control signals for Stage1.
        """
        uncertainty = (entropy + variance) / 2

        # temperature scaling
        temperature = self.max_temp * uncertainty + self.min_temp * (1 - uncertainty)

        # convert strategy → weights
        if confidence < 0.3:
            prompt_weights = {
                "original": 0.2,
                "structured": 0.3,
                "reasoning": 0.3,
                "example": 0.2
            }
        elif confidence < 0.7:
            prompt_weights = {
                "original": 0.3,
                "structured": 0.25,
                "reasoning": 0.25,
                "example": 0.2
            }
        else:
            prompt_weights = {
                "original": 0.5,
                "structured": 0.2,
                "reasoning": 0.2,
                "example": 0.1
            }

        return {
            "temperature": float(temperature),
            "prompt_weights": prompt_weights
        }

    # Stage2 Control
    def _control_stage2(self, confidence, entropy):
        """
        Generate executable control parameters for Stage2.
        """
        rejection_scale = 1.0 + (1.0 - confidence)
        reweight_iter = int(2 + 3 * (1.0 - confidence))
        coupling_lambda = entropy

        return {
            "rejection_scale": float(rejection_scale),
            "reweight_iterations": int(reweight_iter),
            "coupling_lambda": float(coupling_lambda)
        }

    # Default Control
    def _default_control(self):
        """
        Provide default control signals for cold start.
        """
        return {
            "stage1": {
                "temperature": 0.7,
                "prompt_weights": {
                    "original": 0.3,
                    "structured": 0.25,
                    "reasoning": 0.25,
                    "example": 0.2
                }
            },
            "stage2": {
                "rejection_scale": 1.0,
                "reweight_iterations": 2,
                "coupling_lambda": 0.5
            },
            "meta": {
                "confidence": 0.5,
                "weight_entropy": 0.5,
                "signal_variance": 0.5
            }
        }

    # Main Controller Interface
    def process(self, stage2_output, stage3_output):
        """
        Generate feedback signals based on Stage2 and Stage3 outputs.
        """
        if not stage2_output or not stage3_output:
            return self._default_control()

        # Safe Extraction
        confidence = stage2_output.get("confidence", 0.5)

        global_info = stage3_output.get("global", {})
        W = global_info.get("weights", [])
        Z = global_info.get("signals", [])

        if len(W) == 0 or len(Z) == 0:
            return self._default_control()

        # Meta Features
        entropy = self._entropy(W)
        variance = self._variance_signal(Z)

        # Control Generation
        stage1_ctrl = self._control_stage1(confidence, entropy, variance)
        stage2_ctrl = self._control_stage2(confidence, entropy)

        return {
            "stage1": stage1_ctrl,
            "stage2": stage2_ctrl,
            "meta": {
                "confidence": float(confidence),
                "weight_entropy": float(entropy),
                "signal_variance": float(variance)
            }
        }