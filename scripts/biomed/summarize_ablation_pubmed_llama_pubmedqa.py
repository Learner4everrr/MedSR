#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_from_row(row):
    for key in ("reward_model", "score", "metrics"):
        value = row.get(key)
        if isinstance(value, dict):
            if "accuracy" in value:
                return float(value["accuracy"])
            if "score" in value:
                return float(value["score"])
    for key in ("acc", "accuracy", "correct"):
        if key in row:
            return float(row[key])
    return None


def answer_from_row(row):
    for key in ("reward_model", "metrics"):
        value = row.get(key)
        if isinstance(value, dict) and "answer" in value:
            return value["answer"]
    return row.get("answer", "")


def parse_step(path):
    match = re.search(r"(\d+)\.jsonl$", path.name)
    return int(match.group(1)) if match else -1


def summarize_run(dataset_dir):
    val_dir = dataset_dir / "validation_generations"
    files = sorted(val_dir.glob("*.jsonl"), key=parse_step)
    summaries = []
    for file in files:
        rows = load_jsonl(file)
        scores = [score_from_row(row) for row in rows]
        scores = [score for score in scores if score is not None]
        summaries.append(
            {
                "step": parse_step(file),
                "n": len(rows),
                "scored": len(scores),
                "accuracy": sum(scores) / len(scores) if scores else float("nan"),
                "file": str(file),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/ablations_pubmed_llama_pubmedqa")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "summary.tsv"
    rows = []
    for dataset_dir in sorted(root.glob("*/*/pubmedqa_labeled")):
        rel = dataset_dir.relative_to(root)
        group = rel.parts[0]
        job = rel.parts[1]
        for summary in summarize_run(dataset_dir):
            rows.append(
                {
                    "group": group,
                    "job": job,
                    **summary,
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("group\tjob\tstep\tn\tscored\taccuracy\tfile\n")
        for row in rows:
            f.write(
                f"{row['group']}\t{row['job']}\t{row['step']}\t{row['n']}\t"
                f"{row['scored']}\t{row['accuracy']:.6f}\t{row['file']}\n"
            )
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
