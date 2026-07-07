#!/usr/bin/env python3
"""Rescore existing prompt-only baseline generations with robust answer extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomed.baseline_scoring import majority_vote, score_baseline_output  # noqa: E402


DEFAULT_BASELINE_ROOTS = [
    ROOT / "outputs/baselines_prompt_only_3methods",
    ROOT / "outputs/baselines_fewshot_direct",
    ROOT / "outputs/baselines_rag_prompt_all_corpora",
    ROOT / "outputs/baselines_fewshot_deepseek_retry_512",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rescore_file(path: Path, write: bool) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        return {}

    changed = 0
    for row in rows:
        dataset = row["dataset"]
        data_source = row.get("data_source", f"biomed_{dataset}")
        targets = [str(x) for x in row.get("targets", [])]
        old_acc = float(row.get("accuracy", 0.0))

        if row.get("baseline") == "sc_cot" and row.get("raw_outputs"):
            answer, scored_output, candidate_answers = majority_vote(dataset, row["raw_outputs"])
            metrics = score_baseline_output(data_source, dataset, scored_output, targets)
            row["output"] = scored_output
            row["candidate_answers"] = candidate_answers
        else:
            metrics = score_baseline_output(data_source, dataset, row.get("output", ""), targets)
            answer = metrics.get("answer", "")
            row["candidate_answers"] = [answer]

        row["answer"] = metrics.get("answer", "")
        row["accuracy"] = float(metrics.get("accuracy", 0.0))
        row["score"] = float(metrics.get("score", 0.0))
        row["format"] = float(metrics.get("format", 0.0))
        if "scored_output" in metrics:
            row["scored_output"] = metrics["scored_output"]
        if row["accuracy"] != old_acc:
            changed += 1

    n = len(rows)
    summary_path = path.parent / "summary.json"
    old_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary = {
        **old_summary,
        "baseline": rows[0].get("baseline", old_summary.get("baseline")),
        "model_tag": rows[0].get("model_tag", old_summary.get("model_tag")),
        "model_path": rows[0].get("model_path", old_summary.get("model_path")),
        "dataset": rows[0].get("dataset", old_summary.get("dataset")),
        "corpus": old_summary.get("corpus", rows[0].get("corpus")),
        "n": n,
        "accuracy": sum(float(row.get("accuracy", 0.0)) for row in rows) / n,
        "correct": int(sum(float(row.get("accuracy", 0.0)) for row in rows)),
        "score": sum(float(row.get("score", 0.0)) for row in rows) / n,
        "format": sum(float(row.get("format", 0.0)) for row in rows) / n,
        "generations_path": str(path),
    }

    if write:
        write_jsonl(path, rows)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    summary["changed"] = changed
    summary["summary_path"] = str(summary_path)
    return summary


def collect_summaries(roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in roots:
        for path in sorted(root.glob("**/summary.json")):
            row = json.loads(path.read_text())
            row["summary_path"] = str(path)
            rows.append(row)
    return rows


def write_baseline_table(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "corpus", "model", "dataset", "n_shots", "n", "acc", "correct", "score", "format", "path"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (str(r.get("baseline")), str(r.get("corpus")), str(r.get("model_tag")), str(r.get("dataset")))):
            writer.writerow(
                {
                    "experiment": row.get("baseline", ""),
                    "corpus": row.get("corpus") or "none",
                    "model": row.get("model_tag", ""),
                    "dataset": row.get("dataset", ""),
                    "n_shots": row.get("n_shots") or "",
                    "n": row.get("n", 0),
                    "acc": f"{float(row.get('accuracy', 0.0)):.6f}",
                    "correct": row.get("correct", 0),
                    "score": f"{float(row.get('score', 0.0)):.6f}",
                    "format": f"{float(row.get('format', 0.0)):.6f}",
                    "path": row.get("summary_path", row.get("generations_path", "")),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="*", default=[str(x) for x in DEFAULT_BASELINE_ROOTS])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-out", default=str(ROOT / "outputs/result_summaries/baselines_all_with_fewshot.tsv"))
    args = parser.parse_args()

    roots = [Path(x) for x in args.roots]
    changed_summaries = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("**/generations.jsonl")):
            summary = rescore_file(path, write=not args.dry_run)
            if summary:
                changed_summaries.append(summary)

    baseline_summaries = collect_summaries([root for root in roots if root.exists()])
    if not args.dry_run:
        write_baseline_table(baseline_summaries, Path(args.summary_out))

    print("path\tacc\tcorrect\tchanged")
    for row in changed_summaries:
        print(f"{row['summary_path']}\t{row['accuracy']:.6f}\t{row['correct']}\t{row['changed']}")


if __name__ == "__main__":
    main()
