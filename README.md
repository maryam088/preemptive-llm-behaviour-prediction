# Preemptive Prediction of LLM Behaviour Using Balanced Conformal Prediction

Final year project, BSc Computer Science, King's College London.
Author: Fathima Maryam Risvi. Supervisor: Dr. Nicola Paoletti

## Overview

Large language models are typically evaluated only after a full response is generated, which wastes compute on outputs that will later be flagged as failures. This project asks whether a model's own hidden states, captured after a single forward pass and before any output token exists, can predict whether it is about to misbehave.

It reproduces Ashok and May (2025), *Language Models Can Predict Their Own Behaviour* (NeurIPS 2025), then introduces four new conformal decision strategies aimed at safety critical deployment, where missing a failure is far more costly than a false alarm.

## Task and setup

* Model: Llama-3.2-3B-Instruct (28 layers, 3072 dimensional hidden states), probed at layer 14
* Dataset: SelfAware benchmark, unanswerable question detection (600 train, 300 test)
* Label design: behavioural correctness, not surface pattern matching. Label 0 means the model got it wrong (hallucinated an answer, or refused a question it should have answered), a harder and more honest target than just detecting the word "unanswerable"
* Base rate: 66.33% majority class. Calibration set: 90 examples (a key limiting factor)

## Key findings

1. Hidden states at layer 14 do geometrically separate correct from misbehaving responses, confirmed without any gradient based optimisation, via Method 5's difference in means direction.
2. Reliable conformal coverage guarantees need calibration sets of 200 or more examples. The 90 used here caused Method 4 to miss its FNR targets and Method 2 to show non monotone accuracy coverage curves.
3. The choice of decision strategy matters more than the choice of confidence level: at alpha 0.91, false alarm rate alone spans from 7% to 85% across methods.
4. Comparing against the original paper (91%+ accuracy at 40 to 97% coverage on 8B to 70B models, 5000+ examples) suggests the 3B model sits below the scale at which this signal becomes practically deployable without more data.

## Future work

* Scale to Llama-3.1-8B to test whether the signal is a scale artefact.
* Expand training data to 2000 or more examples.
* Inspect the high magnitude dimensions of the Method 5 direction vector for interpretability.
* Sweep across layers rather than fixing layer 14.

## Tech stack

Python, HuggingFace Transformers and Datasets, scikit-learn, NumPy, pandas, matplotlib/seaborn, run on a single Google Colab T4 GPU, orchestrating the original paper's own inference and labelling scripts as subprocesses alongside custom conformal calibration logic.

## References

* Ashok, D. and May, J. (2025). *Language Models Can Predict Their Own Behavior*. NeurIPS 2025. [arXiv:2502.13329](https://doi.org/10.48550/arXiv.2502.13329)
* Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction*. [arXiv:2406.11717](https://doi.org/10.48550/arXiv.2406.11717)
* Marks and Tegmark (2024). *The Geometry of Truth*. COLM 2024. [arXiv:2310.06824](https://doi.org/10.48550/arXiv.2310.06824)
