import argparse
import os

import pandas as pd


SYSTEM = "You are a biomedical QA assistant. You can reason, search, and answer in the required format."


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, tuple) else [value]


def target_list(reward_model):
    gt = reward_model.get("ground_truth", {}) if isinstance(reward_model, dict) else {}
    target = gt.get("target", gt)
    return [str(x) for x in as_list(target)]


def _option_value(opt):
    if isinstance(opt, dict):
        return str(opt.get("value", opt.get("atext", opt))).strip()
    return str(opt).strip()


def _context_lines(context):
    if not isinstance(context, dict):
        return []
    contexts = as_list(context.get("contexts"))
    return [str(x).strip() for x in contexts if str(x).strip()]


def build_question(row):
    question = str(row.get("question") or row.get("qtext") or "").strip()
    context = _context_lines(row.get("context"))
    if context:
        question += "\nAbstract context:\n" + "\n".join(context)

    options = row.get("options")
    if options is None:
        options = row.get("answers")
    if options is not None:
        opts = as_list(options)
        if opts:
            option_lines = []
            for i, opt in enumerate(opts):
                label = str(opt.get("key", "")).strip() if isinstance(opt, dict) else ""
                if not label:
                    label = chr(ord("A") + i)
                option_lines.append(f"{label}. {_option_value(opt)}")
            question += "\nOptions:\n" + "\n".join(option_lines)
    return question


def build_prompt(dataset, question):
    if dataset == "pubmedqa_labeled":
        fmt = "yes, no, or maybe"
        ex = "<answer> yes </answer>"
    else:
        fmt = "one option letter"
        ex = "<answer> A </answer>"
    return (
        "Answer the biomedical question.\n"
        "Put reasoning inside <think>...</think>.\n"
        "First, write one useful search query using key medical terms from the question inside <search>...</search>.\n"
        "The search query must be specific to this question; do not write explanations, URLs, placeholders, or generic text.\n"
        "Do not answer before search results are returned.\n"
        "Search results will appear inside <information>...</information>; then use them to answer.\n"
        f"Give the final answer inside <answer>...</answer>. The answer must be {fmt}; for example {ex}.\n\n"
        f"Question: {question}"
    )


def convert_file(src, dst, dataset, split):
    df = pd.read_parquet(src)
    rows = []
    for idx, row in df.iterrows():
        item = row.to_dict()
        question = build_question(item)
        search_query = str(item.get("question") or item.get("qtext") or question).strip()
        reward_model = item.get("reward_model") or {"style": "rule", "ground_truth": {"target": []}}
        targets = target_list(reward_model)
        prompt = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(dataset, question)},
        ]
        rows.append(
            {
                "data_source": f"biomed_{dataset}",
                "prompt": prompt,
                "ability": "biomedical_qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": targets}},
                "extra_info": {
                    "index": int(item.get("extra_info", {}).get("index", idx)) if isinstance(item.get("extra_info"), dict) else int(idx),
                    "split": split,
                    "question": question,
                    "search_query": search_query,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "search": {
                            "create_kwargs": {
                                "question": question,
                                "ground_truth": {"target": targets},
                                "data_source": f"biomed_{dataset}",
                            }
                        }
                    },
                    "tool_selection": ["search"],
                },
            }
        )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    print(f"wrote {len(rows)} rows: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["medqa_usmle", "headqa", "pubmedqa_labeled"])
    args = parser.parse_args()

    for dataset in args.datasets:
        for split in ["train", "test"]:
            convert_file(
                os.path.join(args.src_root, dataset, f"{split}.parquet"),
                os.path.join(args.dst_root, dataset, f"{split}.parquet"),
                dataset,
                split,
            )


if __name__ == "__main__":
    main()
