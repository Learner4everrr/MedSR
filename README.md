# MedSR

MedSR is a medical search-and-reasoning framework for fine-tuning compact language models to generate structured trajectories containing reasoning, search, retrieved evidence, and constrained final answers.

This repository contains the main code used for the paper experiments:

- `biomed/`: medical QA preprocessing, reward computation, baseline scoring, and search-tool integration.
- `scripts/biomed/`: experiment launch scripts for GRPO fine-tuning, prompt-only baselines, search baselines, retrieval servers, and ablations.
- `config/biomed/`: search-tool configuration.
- `search_r1/`: search/tool-use rollout components adapted for MedSR.
- `verl/`: the local verl-based GRPO training code used by the experiments.
- `paper_scripts/`: scripts used to regenerate summary tables and figures from saved evaluation records.

## Code Base and Attribution

The search/RAG environment and tool-use trajectory infrastructure were adapted from Search-R1: https://github.com/PeterGriffinJin/Search-R1. MedSR builds on this search-and-reasoning setup and extends it to medical question answering with medical QA preprocessing, MedRAG-style retrieval corpora, dataset-specific answer constraints, rule-based medical QA rewards, baselines, ablations, and paper-specific evaluation scripts.

## Installation

Create a Python environment, install the environment requirements, and install the repository in editable mode:

```bash
conda create -n medsr python=3.11 -y
conda activate medsr
pip install -r requirements.txt
pip install -e .
```

The training scripts expect a CUDA-enabled environment with PyTorch, vLLM, Ray, PEFT/LoRA support, and the local retrieval dependencies. If your cluster already provides these packages, `requirements.txt` can be used as a reference for the main Python dependencies.

## Data

Datasets and retrieval corpora are not included in this repository. They are available from their public project pages:

- MedQA-USMLE: https://github.com/jind11/MedQA
- HeadQA: https://github.com/aghie/head-qa
- PubMedQA: https://pubmedqa.github.io/
- MedRAG retrieval corpora: https://huggingface.co/MedRAG

The scripts assume processed datasets, retrieval corpora, and indexes are prepared locally under paths configured by the user.

## Example Usage

Prepare medical QA datasets after downloading the source datasets:

```bash
bash scripts/biomed/prepare_data.sh
```

Launch a MedRAG-style retrieval server. The example below uses the PubMed corpus and the default E5 retriever; update `CORPUS_PATH` and `INDEX_DIR` if your processed corpora and FAISS indexes are stored elsewhere.

```bash
CORPUS_NAME=pubmed \
CORPUS_PATH=data/biomed_corpora/medrag_pubmed/corpus.jsonl \
INDEX_DIR=data/biomed_indexes/medrag_pubmed \
bash scripts/biomed/launch_pubmed_retriever.sh
```

Run the main PubMed-corpus GRPO fine-tuning experiments for the three models and three medical QA datasets:

```bash
bash scripts/biomed/run_formal_pubmed_native_lora_3models_9datasets.sh
```

Run prompt-only baselines:

```bash
bash scripts/biomed/run_prompt_only_baselines_3methods.sh
bash scripts/biomed/run_fewshot_direct_baselines.sh
```

Run untrained search-prompt baselines across corpora:

```bash
bash scripts/biomed/run_rag_prompt_baseline_all_corpora.sh
```

Run the PubMedQA ablation suite used in the paper:

```bash
bash scripts/biomed/run_ablation_pubmed_llama_pubmedqa.sh
```

Regenerate paper result tables and summary figures from saved evaluation outputs:

```bash
python paper_scripts/build_results_assets.py
```

## Reproducibility Notes

The repository intentionally excludes generated outputs, checkpoints, parquet files, FAISS indexes, local environments, cache directories, and W&B logs. Experiment scripts in `scripts/biomed/` document the model, dataset, corpus, retrieval, and fine-tuning settings used for the main experiments and ablations.
