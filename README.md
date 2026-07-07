# MedSR

MedSR is a medical search-and-reasoning framework for fine-tuning compact language models to generate structured trajectories containing reasoning, search, retrieved evidence, and constrained final answers.

This repository contains the main code used for the paper experiments:

- `biomed/`: medical QA preprocessing, reward computation, baseline scoring, and search-tool integration.
- `scripts/biomed/`: experiment launch scripts for GRPO fine-tuning, prompt-only baselines, search baselines, retrieval servers, and ablations.
- `config/biomed/`: search-tool configuration.
- `search_r1/`: search/tool-use rollout components adapted for MedSR.
- `verl/`: the local verl-based GRPO training code used by the experiments.
- `paper_scripts/`: scripts used to regenerate summary tables and figures from saved evaluation records.

## Data

Datasets and retrieval corpora are not included in this repository. They are available from their public project pages:

- MedQA-USMLE: https://github.com/jind11/MedQA
- HeadQA: https://github.com/aghie/head-qa
- PubMedQA: https://pubmedqa.github.io/
- MedRAG retrieval corpora: https://huggingface.co/MedRAG

The scripts assume processed datasets, retrieval corpora, and indexes are prepared locally under paths configured by the user.

## Reproducibility Notes

The repository intentionally excludes generated outputs, checkpoints, parquet files, FAISS indexes, local environments, cache directories, and W&B logs. Experiment scripts in `scripts/biomed/` document the model, dataset, corpus, retrieval, and fine-tuning settings used for the main experiments and ablations.
