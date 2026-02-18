# Phase 3: LLM Review Coverage Analysis

Generated: 2026-02-17T22:56:31.292435+00:00

## Overview

This report measures how well LLM-generated reviews cover the key points
that human reviewers emphasized in their reviews.

## Overall Statistics

- Total papers evaluated: 199
- Total model-paper comparisons: 1000
- Mean keyword recall: 0.420
- Mean F1 score: 0.400

## Per-Model Performance

| Model | N | Exact Recall | Fuzzy Recall | Precision | F1 |
|-------|---|--------------|--------------|-----------|----|
| Qwen2.5-7B-Instruct-Turbo | 200 | 0.125 | 0.426 | 0.443 | 0.418 |
| DeepSeek-V3 | 200 | 0.141 | 0.517 | 0.505 | 0.492 |
| Llama-3.3-70B-Instruct-Turbo | 200 | 0.070 | 0.272 | 0.262 | 0.257 |
| Meta-Llama-3.1-8B-Instruct-Tur | 200 | 0.117 | 0.438 | 0.411 | 0.409 |
| Mixtral-8x7B-Instruct-v0.1 | 200 | 0.105 | 0.450 | 0.441 | 0.425 |

## Model Rankings (by F1 Score)

1. **DeepSeek-V3**: F1 = 0.492
2. **Mixtral-8x7B-Instruct-v0.1**: F1 = 0.425
3. **Qwen2.5-7B-Instruct-Turbo**: F1 = 0.418
4. **Meta-Llama-3.1-8B-Instruct-Turbo**: F1 = 0.409
5. **Llama-3.3-70B-Instruct-Turbo**: F1 = 0.257

## Most Frequently Missed Keywords

These human-reviewer keywords were most often NOT covered by LLMs:

- **ablation study**: missed 191 times
- **experimental valida**: missed 55 times
- **theoretical justifica**: missed 39 times
- **hyperparameter sensitivity**: missed 36 times
- **ablation studie**: missed 34 times
- **statistical significance**: missed 29 times
- **empirical evalua**: missed 26 times
- **ablation study miss**: missed 25 times
- **theoretical analysi**: missed 21 times
- **empirical valida**: missed 20 times

## Key Findings

- **Best performing model**: DeepSeek-V3 (F1=0.492)
- **Lowest performing model**: Llama-3.3-70B-Instruct-Turbo (F1=0.257)
- **Performance gap**: 0.235
