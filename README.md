# Preemptive Prediction of LLM Behaviour Using Balanced Conformal Prediction

Final year project, King's College London. Supervised by Dr. Nicola Paoletti.

## Overview
Detects whether Llama-3.2-3B-Instruct is about to misbehave (e.g. hallucinate an 
answer to an unanswerable question) using only its hidden states from a single 
forward pass — before any output is generated. Replicates and extends Ashok & May 
(2025), "Language Models Can Predict Their Own Behaviour."

## Problem
Standard conformal prediction on probe outputs achieves >90% accuracy, but only 
on the ~10% of inputs the probe is confident enough to commit to at strict alpha 
levels — not a practical early-warning system.

## Approach
- Hidden-state probing on the SelfAware benchmark (unanswerable question detection)
- Linear probe + conformal calibration
- Five decision strategies (M1–M5): baseline, CP threshold, singleton-based rules, 
  conformal risk control, and a geometric difference-in-means classifier

## Tech stack
Python, PyTorch, HuggingFace Transformers, scikit-learn, NumPy

## Future work
Scale to Llama-3.1-8B, combine M5's geometry with M3b's decision rule, 
larger calibration set for conformal risk control.
