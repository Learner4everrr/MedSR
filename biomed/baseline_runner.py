#!/usr/bin/env python3
"""Prompt-only biomedical QA baselines.

This runner is intentionally separate from the GRPO training code. It supports:
  - direct: direct answer without retrieval
  - direct_1shot/direct_3shot/direct_5shot/direct_10shot: few-shot direct answer without retrieval
  - cot: chain-of-thought answer without retrieval
  - sc_cot: self-consistency chain-of-thought without retrieval
  - rag_prompt: retrieve top-k documents first, then answer without training
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PATH"] = f"{ROOT / '.venv' / 'bin'}:{os.environ.get('PATH', '')}"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from biomed.baseline_scoring import (  # noqa: E402
    extract_final_answer,
    majority_vote,
    score_baseline_output,
)


DATASETS = ("medqa_usmle", "headqa", "pubmedqa_labeled")
BASELINES = (
    "direct",
    "direct_1shot",
    "direct_3shot",
    "direct_5shot",
    "direct_10shot",
    "cot",
    "sc_cot",
    "rag_prompt",
)
SYSTEM = "You are a biomedical QA assistant."


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return [value]


def targets_from_reward_model(reward_model: Any) -> list[str]:
    if not isinstance(reward_model, dict):
        return []
    ground_truth = reward_model.get("ground_truth", {})
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", ground_truth.get("answer", []))
    else:
        target = ground_truth
    return [str(x) for x in as_list(target)]


def answer_format(dataset: str) -> tuple[str, str]:
    if dataset == "pubmedqa_labeled":
        return "yes, no, or maybe", "<answer>yes</answer>"
    return "one option letter such as A, B, C, D, or E", "<answer>A</answer>"


def question_from_row(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"]).strip()
    prompt = as_list(row.get("prompt"))
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = str(msg.get("content", ""))
            marker = "Question:"
            if marker in content:
                return content.split(marker, 1)[1].strip()
            return content.strip()
    return ""


def search_query_from_row(row: dict[str, Any], question: str) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("search_query"):
        return str(extra["search_query"]).strip()
    return question.split("\nOptions:", 1)[0].split("\nAbstract context:", 1)[0].strip()


def shot_count(baseline: str) -> int:
    match = re.fullmatch(r"direct_(\d+)shot", baseline)
    if match:
        return int(match.group(1))
    return 0


def build_fewshot_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    blocks = ["Examples:"]
    for i, example in enumerate(examples, start=1):
        target = str(as_list(example["targets"])[0]).strip()
        blocks.append(
            f"Example {i}\n"
            f"Question:\n{example['question']}\n"
            f"Answer:\n<answer>{target}</answer>"
        )
    return "\n\n".join(blocks) + "\n\n"


def compact_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "question": str(example.get("question", "")),
            "targets": [str(x) for x in as_list(example.get("targets", []))],
            "data_source": str(example.get("data_source", "")),
        }
        for example in examples
    ]


def build_messages(
    dataset: str,
    question: str,
    baseline: str,
    evidence: str | None = None,
    fewshot_examples: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    fmt, example = answer_format(dataset)
    base = (
        "Answer the biomedical question.\n"
        f"The final answer must be {fmt}.\n"
        f"Put only the final answer inside <answer>...</answer>, for example {example}.\n"
    )
    if baseline == "direct" or shot_count(baseline):
        instruction = base + "Do not include reasoning.\n"
    elif baseline in {"cot", "sc_cot"}:
        instruction = base + "Reason briefly inside <think>...</think>, then give the final answer.\n"
    elif baseline == "rag_prompt":
        instruction = (
            base
            + "Use the retrieved evidence if it is helpful. "
            + "Reason briefly inside <think>...</think>, then give the final answer.\n"
            + f"\nRetrieved evidence:\n{evidence or 'No retrieved evidence.'}\n"
        )
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    fewshot = build_fewshot_block(fewshot_examples or [])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{instruction}\n{fewshot}Question:\n{question}"},
    ]


def render_prompts(tokenizer: AutoTokenizer, messages: list[list[dict[str, str]]]) -> list[str]:
    rendered = []
    for item in messages:
        rendered.append(tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True))
    return rendered


def doc_text(doc: Any, max_chars: int) -> str:
    if isinstance(doc, dict):
        if "document" in doc and isinstance(doc["document"], (dict, str)):
            return doc_text(doc["document"], max_chars)
        title = doc.get("title") or doc.get("id") or ""
        text = doc.get("contents") or doc.get("content") or doc.get("text") or str(doc)
        if title:
            return f"{title}: {str(text)[:max_chars]}"
        return str(text)[:max_chars]
    return str(doc)[:max_chars]


def retrieve_evidence(queries: list[str], url: str, topk: int, max_doc_chars: int, batch_size: int) -> list[str]:
    all_evidence: list[str] = []
    for start in tqdm(range(0, len(queries), batch_size), desc="retrieving"):
        batch = queries[start : start + batch_size]
        response = requests.post(
            url,
            json={"queries": batch, "topk": topk, "return_scores": True},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        docs = payload.get("result", payload)
        if isinstance(docs, dict):
            docs = docs.get("results", docs.get("documents", docs))
        if len(batch) == 1 and isinstance(docs, list) and (not docs or isinstance(docs[0], dict)):
            docs = [docs]
        for per_query in docs:
            lines = []
            for i, doc in enumerate(as_list(per_query)[:topk], start=1):
                lines.append(f"Doc {i}: {doc_text(doc, max_doc_chars)}")
            all_evidence.append("\n".join(lines))
    if len(all_evidence) != len(queries):
        raise RuntimeError(f"Retriever returned {len(all_evidence)} groups for {len(queries)} queries")
    return all_evidence


def load_rows(data_root: Path, dataset: str, split: str, max_examples: int | None) -> list[dict[str, Any]]:
    path = data_root / dataset / f"{split}.parquet"
    df = pd.read_parquet(path)
    if max_examples is not None:
        df = df.head(max_examples)
    return [row.to_dict() for _, row in df.iterrows()]


def build_records(rows: list[dict[str, Any]], dataset: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records = []
    questions = []
    queries = []
    for i, row in enumerate(rows):
        question = question_from_row(row)
        questions.append(question)
        queries.append(search_query_from_row(row, question))
        records.append(
            {
                "idx": i,
                "data_source": str(row.get("data_source", f"biomed_{dataset}")),
                "question": question,
                "targets": targets_from_reward_model(row.get("reward_model")),
                "extra_info": row.get("extra_info", {}),
            }
        )
    return records, questions, queries


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    if args.baseline not in BASELINES:
        raise ValueError(f"--baseline must be one of {BASELINES}")
    if args.dataset not in DATASETS:
        raise ValueError(f"--dataset must be one of {DATASETS}")

    rows = load_rows(Path(args.data_root), args.dataset, args.split, args.max_examples)
    records, questions, queries = build_records(rows, args.dataset)

    fewshot_examples: list[dict[str, Any]] = []
    n_shots = shot_count(args.baseline)
    if n_shots:
        train_rows = load_rows(Path(args.data_root), args.dataset, args.fewshot_split, n_shots)
        train_records, _, _ = build_records(train_rows, args.dataset)
        fewshot_examples = [
            record for record in train_records if record["question"] and record["targets"]
        ][:n_shots]
        if len(fewshot_examples) != n_shots:
            raise RuntimeError(f"Only found {len(fewshot_examples)} usable few-shot examples for {args.dataset}")

    evidence = [None] * len(records)
    if args.baseline == "rag_prompt":
        evidence = retrieve_evidence(
            queries,
            args.retriever_url,
            args.topk,
            args.max_doc_chars,
            args.retrieve_batch_size,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    messages = [
        build_messages(args.dataset, record["question"], args.baseline, evidence[i], fewshot_examples)
        for i, record in enumerate(records)
    ]
    prompts = render_prompts(tokenizer, messages)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        download_dir=args.download_dir,
        disable_log_stats=True,
    )

    if args.baseline == "direct" or shot_count(args.baseline):
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.direct_max_tokens)
    elif args.baseline == "cot":
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.cot_max_tokens)
    elif args.baseline == "sc_cot":
        sampling = SamplingParams(
            temperature=args.sc_temperature,
            top_p=args.sc_top_p,
            n=args.sc_samples,
            max_tokens=args.cot_max_tokens,
        )
    else:
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.cot_max_tokens)

    request_outputs = llm.generate(prompts, sampling)
    out_rows = []
    for record, prompt_text, request_output in zip(records, prompts, request_outputs):
        candidate_outputs = [o.text for o in request_output.outputs]
        if args.baseline == "sc_cot":
            answer, scored_output, candidate_answers = majority_vote(args.dataset, candidate_outputs)
            metrics = score_baseline_output(record["data_source"], args.dataset, scored_output, record["targets"])
            output = scored_output
        else:
            output = candidate_outputs[0] if candidate_outputs else ""
            answer = extract_final_answer(args.dataset, output)
            candidate_answers = [answer]
            metrics = score_baseline_output(record["data_source"], args.dataset, output, record["targets"])
        out_rows.append(
            {
                "idx": record["idx"],
                "baseline": args.baseline,
                "model_tag": args.model_tag,
                "model_path": args.model_path,
                "dataset": args.dataset,
                "data_source": record["data_source"],
                "question": record["question"],
                "targets": record["targets"],
                "prompt": prompt_text,
                "output": output,
                "raw_outputs": candidate_outputs if args.baseline == "sc_cot" else None,
                "fewshot_examples": compact_examples(fewshot_examples) if n_shots else None,
                "candidate_answers": candidate_answers,
                "answer": answer,
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "score": float(metrics.get("score", 0.0)),
                "format": float(metrics.get("format", 0.0)),
                "retrieved_evidence": evidence[record["idx"]] if args.baseline == "rag_prompt" else None,
            }
        )

    out_dir = Path(args.output_root) / args.baseline
    if args.corpus:
        out_dir = out_dir / args.corpus
    out_dir = out_dir / args.model_tag / args.dataset
    generations_path = out_dir / "generations.jsonl"
    summary_path = out_dir / "summary.json"
    write_jsonl(generations_path, out_rows)

    n = len(out_rows)
    summary = {
        "baseline": args.baseline,
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "corpus": args.corpus,
        "split": args.split,
        "fewshot_split": args.fewshot_split if n_shots else None,
        "n_shots": n_shots,
        "n": n,
        "accuracy": sum(row["accuracy"] for row in out_rows) / n if n else 0.0,
        "correct": int(sum(row["accuracy"] for row in out_rows)),
        "score": sum(row["score"] for row in out_rows) / n if n else 0.0,
        "format": sum(row["format"] for row in out_rows) / n if n else 0.0,
        "retriever_url": args.retriever_url if args.baseline == "rag_prompt" else None,
        "topk": args.topk if args.baseline == "rag_prompt" else None,
        "max_doc_chars": args.max_doc_chars if args.baseline == "rag_prompt" else None,
        "generations_path": str(generations_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--fewshot-split", default="train")
    parser.add_argument("--data-root", default=str(ROOT / "data/biomed"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs/baselines_prompt_only"))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--download-dir", default=os.getenv("HF_HUB_CACHE"))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direct-max-tokens", type=int, default=64)
    parser.add_argument("--cot-max-tokens", type=int, default=512)
    parser.add_argument("--sc-samples", type=int, default=5)
    parser.add_argument("--sc-temperature", type=float, default=0.7)
    parser.add_argument("--sc-top-p", type=float, default=0.95)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    parser.add_argument("--retrieve-batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(parse_args())
