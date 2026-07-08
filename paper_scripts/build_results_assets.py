#!/usr/bin/env python3
"""Build result tables for the npj Digital Medicine draft."""

from __future__ import annotations

import json
import math
import string
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers/v1"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
MAIN = ROOT / "outputs/result_summaries/main_grpo.tsv"
BASE = ROOT / "outputs/result_summaries/baselines_all_with_fewshot.tsv"
ABL = ROOT / "outputs/ablations_pubmed_llama_pubmedqa/summary.tsv"

MODEL_LABELS = {
    "qwen25_3b": "Qwen2.5-3B",
    "llama32_3b_instruct": "Llama-3.2-3B-Instruct",
    "deepseek_r1_qwen_15b": "DeepSeek-R1-Distill-Qwen-1.5B",
}
MODEL_SHORT = {
    "qwen25_3b": "Qwen",
    "llama32_3b_instruct": "Llama",
    "deepseek_r1_qwen_15b": "DeepSeek",
}
FIGURE_MODEL_LABELS = {
    "qwen25_3b": "Qwen2.5-3B",
    "llama32_3b_instruct": "Llama-3.2-3B\nInstruct",
    "deepseek_r1_qwen_15b": "DeepSeek-R1\nQwen-1.5B",
}
DATASET_LABELS = {
    "medqa_usmle": "MedQA",
    "headqa": "HeadQA",
    "pubmedqa_labeled": "PubMedQA",
}
CORPUS_LABELS = {
    "pubmed": "PubMed",
    "textbooks": "Textbooks",
    "statpearls": "StatPearls",
    "wikipedia": "Wikipedia",
    "none": "-",
}
CORPUS_SHORT = {
    "pubmed": "PubMed",
    "textbooks": "Textbk",
    "statpearls": "StatP",
    "wikipedia": "Wiki",
}
NATURE_COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#4D4D4D",
}
NATURE_SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "nature_sequential_blue",
    ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
)
PANEL_TITLE_SIZE = 16
PANEL_TITLE_PAD = 5


def pct(x: float | int | str) -> str:
    return f"{100 * float(x):.1f}"


def metric_pct(x: float | int | str) -> str:
    return f"{100 * float(x):.1f}"


def format_pct(x: float | int | str) -> str:
    value = 100 * float(x)
    return "100" if abs(value - 100.0) < 0.05 else f"{value:.1f}"


def tex_escape(text: object) -> str:
    return str(text).replace("&", "\\&").replace("_", "\\_")


def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value)
    table = str.maketrans("", "", string.punctuation)
    return "".join(text.lower().translate(table).split())


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def f1_metrics(golds: list[str], preds: list[str]) -> dict[str, float]:
    labels = sorted(set(golds) | set(preds))
    if not labels:
        return {
            "micro_f1": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
        }

    total_tp = total_fp = total_fn = 0
    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    weighted_sum = 0.0
    total_support = 0
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        support = sum(g == label for g in golds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_sum += support * f1
        total_support += support

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "micro_f1": micro_f1,
        "macro_f1": sum(f1s) / len(f1s),
        "weighted_f1": weighted_sum / total_support if total_support else 0.0,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
    }


def generations_path(path_value: object, is_baseline: bool) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    if is_baseline:
        return path.with_name("generations.jsonl")
    return path


def load_generation_metrics(path: Path, fallback: pd.Series) -> dict[str, float | int | str]:
    golds: list[str] = []
    preds: list[str] = []
    correct = 0
    fmt_values: list[float] = []
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                targets = item.get("targets")
                if targets is None:
                    targets = item.get("gts", {}).get("target", [])
                if not isinstance(targets, list):
                    targets = [targets]
                norm_targets = [normalize_answer(target) for target in targets]
                pred = normalize_answer(item.get("answer", ""))
                gold = pred if pred in norm_targets else (norm_targets[0] if norm_targets else "")
                golds.append(gold)
                preds.append(pred or "__invalid__")
                correct += int(pred in norm_targets)
                if "format" in item:
                    try:
                        fmt_values.append(float(item["format"]))
                    except (TypeError, ValueError):
                        pass
    else:
        total = int(fallback.get("n", 0))
        correct = int(fallback.get("correct", round(float(fallback.get("acc", 0.0)) * total)))
        golds = ["correct"] * correct + ["wrong"] * (total - correct)
        preds = ["correct"] * correct + ["incorrect"] * (total - correct)

    total = len(golds)
    acc = correct / total if total else float("nan")
    ci_low, ci_high = wilson_ci(correct, total)
    metrics = f1_metrics(golds, preds)
    metrics.update(
        {
            "n": total,
            "correct": correct,
            "acc": acc,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "format": sum(fmt_values) / len(fmt_values) if fmt_values else float(fallback.get("format", float("nan"))),
        }
    )
    return metrics


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(MAIN, sep="\t"), pd.read_csv(BASE, sep="\t"), pd.read_csv(ABL, sep="\t")


def acc_cell(df: pd.DataFrame, **filters: str) -> str:
    sub = df
    for key, value in filters.items():
        sub = sub[sub[key] == value]
    if sub.empty:
        return "--"
    return pct(sub.iloc[0].acc)


def best_cell(df: pd.DataFrame, suffix_col: str | None = None) -> tuple[str, float]:
    if df.empty:
        return "--", float("nan")
    row = df.loc[df.acc.idxmax()]
    suffix = ""
    if suffix_col is not None:
        value = row[suffix_col]
        if pd.notna(value) and str(value) != "":
            if suffix_col == "corpus":
                suffix = f" ({CORPUS_SHORT.get(str(value), str(value))})"
            else:
                suffix = f" ({int(float(value))})"
    return f"{pct(row.acc)}{suffix}", float(row.acc)


def experiment_label(row: pd.Series, source: str) -> str:
    exp = str(row.experiment)
    if source == "Trained":
        return "Ours"
    if exp == "direct":
        return "Direct"
    if exp == "cot":
        return "CoT"
    if exp == "sc_cot":
        return "SC-CoT"
    if exp == "rag_prompt":
        return "Search prompt"
    if exp.startswith("direct_") and exp.endswith("shot"):
        shots = str(row.get("n_shots", "")).replace(".0", "")
        return f"{shots}-shot"
    return exp


def add_metric_rows(df: pd.DataFrame, source: str, is_baseline: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sort_cols = ["dataset", "model", "experiment", "corpus"]
    for _, row in df.sort_values(sort_cols).iterrows():
        path = generations_path(row.path, is_baseline)
        metrics = load_generation_metrics(path, row)
        rows.append(
            {
                "dataset": str(row.dataset),
                "model": str(row.model),
                "source": source,
                "experiment": experiment_label(row, source),
                "corpus": str(row.corpus),
                "n": int(metrics["n"]),
                "correct": int(metrics["correct"]),
                "acc": float(metrics["acc"]),
                "ci_low": float(metrics["ci_low"]),
                "ci_high": float(metrics["ci_high"]),
                "micro_f1": float(metrics["micro_f1"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "format": float(metrics["format"]),
                "path": str(path),
            }
        )
    return rows


def build_main_metrics(main: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = add_metric_rows(base, "Baseline", True) + add_metric_rows(main, "Trained", False)
    metrics = pd.DataFrame(rows)
    dataset_order = {name: idx for idx, name in enumerate(DATASET_LABELS)}
    model_order = {name: idx for idx, name in enumerate(MODEL_LABELS)}
    source_order = {"Baseline": 0, "Trained": 1}
    exp_order = {
        "Direct": 0,
        "CoT": 1,
        "SC-CoT": 2,
        "1-shot": 3,
        "3-shot": 4,
        "5-shot": 5,
        "10-shot": 6,
        "Search prompt": 7,
        "GRPO": 8,
    }
    corpus_order = {"none": 0, "pubmed": 1, "textbooks": 2, "statpearls": 3, "wikipedia": 4}
    metrics["_dataset_order"] = metrics.dataset.map(dataset_order)
    metrics["_model_order"] = metrics.model.map(model_order)
    metrics["_source_order"] = metrics.source.map(source_order)
    metrics["_exp_order"] = metrics.experiment.map(exp_order)
    metrics["_corpus_order"] = metrics.corpus.map(corpus_order)
    return metrics.sort_values(
        ["_dataset_order", "_model_order", "_source_order", "_exp_order", "_corpus_order"]
    ).drop(columns=["_dataset_order", "_model_order", "_source_order", "_exp_order", "_corpus_order"])


def ci_cell(row: pd.Series) -> str:
    return f"{metric_pct(row.ci_low)}--{metric_pct(row.ci_high)}"


def bold_if_best(text: str, is_best: bool) -> str:
    return rf"\textbf{{{text}}}" if is_best else text


def dataset_metric_table_rows(metrics: pd.DataFrame, dataset_key: str) -> list[str]:
    dataset_metrics = metrics[metrics.dataset == dataset_key]
    dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)
    row_order = (
        dataset_metrics[["experiment", "corpus"]]
        .drop_duplicates()
        .sort_values(
            by=["experiment", "corpus"],
            key=lambda col: col.map(
                {
                    "Direct": 0,
                    "CoT": 1,
                    "SC-CoT": 2,
                    "1-shot": 3,
                    "3-shot": 4,
                    "5-shot": 5,
                    "10-shot": 6,
                    "Search prompt": 7,
                    "GRPO": 8,
                    "none": 0,
                    "pubmed": 1,
                    "textbooks": 2,
                    "statpearls": 3,
                    "wikipedia": 4,
                }
            ),
        )
    )
    metric_lookup = {
        (str(row.model), str(row.experiment), str(row.corpus)): row
        for _, row in dataset_metrics.iterrows()
    }
    best_flags: dict[tuple[str, str, str, str], bool] = {}
    best_metric_cols = ["acc", "micro_f1", "macro_f1", "weighted_f1", "format"]
    for model in MODEL_LABELS:
        model_metrics = dataset_metrics[dataset_metrics.model == model]
        for metric_col in best_metric_cols:
            if model_metrics.empty:
                continue
            best_value = model_metrics[metric_col].max()
            for _, row in model_metrics.iterrows():
                key = (model, str(row.experiment), str(row.corpus), metric_col)
                best_flags[key] = pd.notna(row[metric_col]) and abs(float(row[metric_col]) - float(best_value)) < 1e-12
                if metric_col == "acc":
                    ci_key = (model, str(row.experiment), str(row.corpus), "ci")
                    best_flags[ci_key] = best_flags[key]

    lines = [
        rf"\rowcolor{{gray!12}} \multicolumn{{20}}{{l}}{{\textbf{{{tex_escape(dataset_label)}}}}} \\",
    ]
    for _, row_key in row_order.iterrows():
        experiment = str(row_key.experiment)
        corpus = str(row_key.corpus)
        row_prefix = r"\rowcolor{blue!5} " if experiment == "Ours" else ""
        values = [rf"\hspace{{0.45em}}{tex_escape(experiment)}", tex_escape(CORPUS_LABELS.get(corpus, corpus))]
        for model in MODEL_LABELS:
            row = metric_lookup.get((model, experiment, corpus))
            if row is None:
                values.extend(["--"] * 6)
                continue
            key_prefix = (model, experiment, corpus)
            values.extend(
                [
                    bold_if_best(metric_pct(row.acc), best_flags.get((*key_prefix, "acc"), False)),
                    bold_if_best(ci_cell(row), best_flags.get((*key_prefix, "ci"), False)),
                    bold_if_best(metric_pct(row.micro_f1), best_flags.get((*key_prefix, "micro_f1"), False)),
                    bold_if_best(metric_pct(row.macro_f1), best_flags.get((*key_prefix, "macro_f1"), False)),
                    bold_if_best(metric_pct(row.weighted_f1), best_flags.get((*key_prefix, "weighted_f1"), False)),
                    bold_if_best(
                        format_pct(row.format) if pd.notna(row.format) else "--",
                        best_flags.get((*key_prefix, "format"), False),
                    ),
                ]
            )
        lines.append(row_prefix + " & ".join(values) + r" \\")
    return lines


def write_main_metric_tables(metrics: pd.DataFrame) -> None:
    metric_headers = ["Acc", "95\\% CI", "Micro", "Macro", "Wt", "Fmt"]
    table_model_labels = {
        "qwen25_3b": "Qwen2.5-3B",
        "llama32_3b_instruct": "Llama-3.2-3B-Instruct",
        "deepseek_r1_qwen_15b": "DeepSeek-R1-Distill-Qwen-1.5B",
    }
    table_corpus_labels = {
        "none": "-",
        "pubmed": "PubMed",
        "textbooks": "Text",
        "statpearls": "StatP",
        "wikipedia": "Wiki",
    }
    table_experiment_labels = {
        "Search prompt": "Search",
    }
    order_map = {
        "Direct": 0,
        "CoT": 1,
        "SC-CoT": 2,
        "1-shot": 3,
        "3-shot": 4,
        "5-shot": 5,
        "10-shot": 6,
        "Search prompt": 7,
        "Ours": 8,
        "none": 0,
        "pubmed": 1,
        "textbooks": 2,
        "statpearls": 3,
        "wikipedia": 4,
    }
    metric_cols = ["acc", "micro_f1", "macro_f1", "weighted_f1", "format"]

    def dataset_rows(dataset_key: str) -> list[str]:
        dataset_metrics = metrics[metrics.dataset == dataset_key]
        dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)
        row_order = (
            dataset_metrics[["experiment", "corpus"]]
            .drop_duplicates()
            .sort_values(
                by=["experiment", "corpus"],
                key=lambda col: col.map(order_map),
            )
        )
        lookup = {
            (str(row.model), str(row.experiment), str(row.corpus)): row
            for _, row in dataset_metrics.iterrows()
        }
        best_flags: dict[tuple[str, str, str, str], bool] = {}
        for model in MODEL_LABELS:
            model_metrics = dataset_metrics[dataset_metrics.model == model]
            best_values = {col: model_metrics[col].max() for col in metric_cols}
            for _, row in model_metrics.iterrows():
                prefix = (model, str(row.experiment), str(row.corpus))
                for col in metric_cols:
                    best_flags[(*prefix, col)] = (
                        pd.notna(row[col]) and abs(float(row[col]) - float(best_values[col])) < 1e-12
                    )
                best_flags[(*prefix, "ci")] = best_flags[(*prefix, "acc")]

        lines: list[str] = []
        lines.append(rf"\rowcolor{{gray!20}} \multicolumn{{20}}{{l}}{{\hspace{{0.25em}}\textbf{{{tex_escape(dataset_label)}}}}} \\")
        for _, row_key in row_order.iterrows():
            experiment = str(row_key.experiment)
            corpus = str(row_key.corpus)
            row_prefix = r"\rowcolor{blue!10} " if experiment == "Ours" else ""
            values = [
                rf"\hspace{{0.45em}}{tex_escape(table_experiment_labels.get(experiment, experiment))}",
                tex_escape(table_corpus_labels.get(corpus, CORPUS_LABELS.get(corpus, corpus))),
            ]
            for model in MODEL_LABELS:
                row = lookup.get((model, experiment, corpus))
                if row is None:
                    values.extend(["--"] * 6)
                    continue
                key = (model, experiment, corpus)
                values.extend(
                    [
                        bold_if_best(metric_pct(row.acc), best_flags.get((*key, "acc"), False)),
                        bold_if_best(ci_cell(row), best_flags.get((*key, "ci"), False)),
                        bold_if_best(metric_pct(row.micro_f1), best_flags.get((*key, "micro_f1"), False)),
                        bold_if_best(metric_pct(row.macro_f1), best_flags.get((*key, "macro_f1"), False)),
                        bold_if_best(metric_pct(row.weighted_f1), best_flags.get((*key, "weighted_f1"), False)),
                        bold_if_best(
                            format_pct(row.format) if pd.notna(row.format) else "--",
                            best_flags.get((*key, "format"), False),
                        ),
                    ]
                )
            lines.append(row_prefix + " & ".join(values) + r" \\")
        return lines

    header = (
        r"\multicolumn{2}{c}{} & "
        + rf"\multicolumn{{6}}{{c}}{{{tex_escape(table_model_labels['qwen25_3b'])}}} & "
        + rf"\multicolumn{{6}}{{c}}{{{tex_escape(table_model_labels['llama32_3b_instruct'])}}} & "
        + rf"\multicolumn{{6}}{{c}}{{{tex_escape(table_model_labels['deepseek_r1_qwen_15b'])}}} \\"
    )
    metric_line = "Experiment & Corpus & " + " & ".join(metric_headers * 3) + r" \\"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{\textbf{Main experimental results across all three medical QA benchmarks.} Rows are grouped by dataset and each row is one evaluated method--corpus condition. ``Search'' denotes an untrained search baseline in which retrieved passages from the named corpus are inserted into the model context without parameter updates. ``Ours'' denotes the GRPO-trained MedSR search-and-answer policy using the same corpus interface. Model blocks report accuracy, Wilson 95\% confidence interval, micro-F1, macro-F1, weighted-F1, and mean format score. Abbreviations: Text, textbooks; StatP, StatPearls; Wiki, Wikipedia; Wt, weighted-F1; Fmt, format score. Within each dataset and model block, the best value for each metric column is bolded.}\label{tab:main-results}",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.0pt}",
        r"\renewcommand{\arraystretch}{0.66}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{\hspace{3pt}}cccccc@{\hspace{3pt}}cccccc@{}}",
        r"\toprule",
        header,
        r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}\cmidrule(lr){15-20}",
        metric_line,
        r"\midrule",
    ]
    for idx, dataset in enumerate(DATASET_LABELS):
        if idx:
            lines.append(r"\midrule")
        lines.extend(dataset_rows(dataset))
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    (TAB / "table_main_results.tex").write_text("\n".join(lines))

    stale_files = [
        TAB / "table_main_experiments.tex",
        TAB / "table_main_medqa_metrics.tex",
        TAB / "table_main_headqa_metrics.tex",
        TAB / "table_main_pubmedqa_metrics.tex",
        TAB / "table_main_medqa_results.tex",
        TAB / "table_main_headqa_results.tex",
        TAB / "table_main_pubmedqa_results.tex",
    ]
    for stale in stale_files:
        if stale.exists():
            stale.unlink()


def write_plain_table(path: Path, header: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def write_main_experiment_table(main: pd.DataFrame, base: pd.DataFrame) -> None:
    metrics = build_main_metrics(main, base)
    write_main_metric_tables(metrics)
    metrics.to_csv(TAB / "main_metrics.csv", index=False)

    for dataset in DATASET_LABELS:
        stale = TAB / f"table_main_{dataset}.tex"
        if stale.exists():
            stale.unlink()


def write_ablation_figure(abl: pd.DataFrame) -> None:
    final = abl.sort_values("step").groupby(["group", "job"], as_index=False).tail(1)

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(12.5, 8.8))
    gs = fig.add_gridspec(2, 3)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1:]),
    ]
    colors = [
        NATURE_COLORS["blue"],
        NATURE_COLORS["orange"],
        NATURE_COLORS["green"],
        NATURE_COLORS["purple"],
        NATURE_COLORS["vermillion"],
        NATURE_COLORS["sky"],
    ]

    def bar_panel(
        ax,
        group: str,
        title: str,
        labels: dict[str, str] | None = None,
        order: list[str] | None = None,
    ) -> None:
        sub = final[final.group == group].sort_values("job")
        if order is not None:
            sub = sub.set_index("job").loc[order].reset_index()
        names = [labels.get(v, v) if labels else v for v in sub.job]
        vals = [100 * float(v) for v in sub.accuracy]
        bars = ax.bar(range(len(vals)), vals, color=colors[: len(vals)], edgecolor="black", linewidth=0.4)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(max(0, min(vals) - 8), min(100, max(vals) + 5))
        ax.set_xticks(range(len(vals)), names, rotation=28, ha="right")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    bar_panel(
        axes[0],
        "topk",
        "(a) Retrieved passages",
        {"k0_no_retrieval": "k=0", "k1": "k=1", "k2": "k=2", "k3": "k=3", "k5": "k=5"},
        ["k0_no_retrieval", "k1", "k2", "k3", "k5"],
    )
    bar_panel(
        axes[1],
        "reward",
        "(b) Reward design",
        {
            "accuracy_only": "answer",
            "answer_plus_format": "answer+format",
            "format_heavy": "format-heavy",
            "length_penalty": "length penalty",
            "search_light": "search-light",
        },
    )
    bar_panel(
        axes[2],
        "retriever",
        "(c) Corpus / retriever",
        {
            "e5_pubmed": "PubMed",
            "e5_statpearls": "StatPearls",
            "e5_textbooks": "Textbooks",
            "e5_wikipedia": "Wikipedia",
        },
    )
    bar_panel(
        axes[3],
        "lora",
        "(d) LoRA rank",
        {"rank4": "r=4", "rank8": "r=8", "rank16": "r=16", "rank32": "r=32", "rank64": "r=64"},
        ["rank4", "rank8", "rank16", "rank32", "rank64"],
    )

    steps = abl[(abl.group == "steps") & (abl.step > 0)].sort_values("step")
    ax = axes[4]
    x = [int(v) for v in steps.step]
    y = [100 * float(v) for v in steps.accuracy]
    ax.plot(x, y, marker="o", linewidth=2.0, color="#4C78A8")
    ax.scatter(x, y, s=35, color="#F58518", edgecolor="black", linewidth=0.4, zorder=3)
    ax.set_title("(e) Training duration", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(65, min(100, max(y) + 4))
    ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.7)
    for step, val in zip(x, y):
        if step in {0, 500, 1000, 1500, 2000}:
            ax.text(step, val + 0.55, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.subplots_adjust(left=0.055, right=0.99, top=0.955, bottom=0.08, wspace=0.28, hspace=0.34)
    fig.savefig(FIG / "fig_ablation_summary.pdf")
    fig.savefig(FIG / "fig_ablation_summary.png", dpi=300)
    plt.close(fig)


def write_main_result_figure(metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    model_keys = list(MODEL_LABELS)
    dataset_keys = list(DATASET_LABELS)
    corpus_keys = ["pubmed", "textbooks", "statpearls", "wikipedia"]
    corpus_colors = {
        "pubmed": NATURE_COLORS["blue"],
        "textbooks": NATURE_COLORS["orange"],
        "statpearls": NATURE_COLORS["green"],
        "wikipedia": NATURE_COLORS["purple"],
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(13.2, 13.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0, 0.92])
    ax_gain = fig.add_subplot(gs[0, 0])
    ax_corpus = fig.add_subplot(gs[0, 1])
    ax_pair = fig.add_subplot(gs[1, 0])
    ax_best = fig.add_subplot(gs[1, 1])
    ax_choice = fig.add_subplot(gs[2, 0])
    ax_delta = fig.add_subplot(gs[2, 1])

    gains: list[list[float]] = []
    annotations: list[list[str]] = []
    for model in model_keys:
        gain_row: list[float] = []
        ann_row: list[str] = []
        for dataset in dataset_keys:
            sub = metrics[(metrics.dataset == dataset) & (metrics.model == model)]
            best_baseline = sub[sub.source == "Baseline"].acc.max()
            trained = sub[sub.source == "Trained"]
            best_idx = trained.acc.idxmax()
            best_trained = trained.loc[best_idx]
            gain = 100 * (float(best_trained.acc) - float(best_baseline))
            gain_row.append(gain)
            ann_row.append(f"{gain:+.1f}\n{CORPUS_SHORT.get(str(best_trained.corpus), str(best_trained.corpus))}")
        gains.append(gain_row)
        annotations.append(ann_row)

    vmax = max(value for row in gains for value in row)
    heat = ax_gain.imshow(gains, cmap=NATURE_SEQUENTIAL_BLUE, vmin=0, vmax=vmax, aspect="auto")
    ax_gain.set_title("(a) Best trained-search gain", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_gain.set_xticks(range(len(dataset_keys)), [DATASET_LABELS[d] for d in dataset_keys])
    ax_gain.set_yticks(range(len(model_keys)), [FIGURE_MODEL_LABELS[m] for m in model_keys])
    ax_gain.set_xlabel("Dataset")
    ax_gain.set_ylabel("Model")
    for i, row in enumerate(annotations):
        for j, text in enumerate(row):
            color = "white" if gains[i][j] > 0.65 * vmax else "black"
            ax_gain.text(j, i, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    cbar = fig.colorbar(heat, ax=ax_gain, fraction=0.046, pad=0.03)
    cbar.set_label("Accuracy gain vs strongest untrained baseline (pp)")

    width = 0.18
    x = range(len(dataset_keys))
    for offset_idx, corpus in enumerate(corpus_keys):
        vals = []
        for dataset in dataset_keys:
            sub = metrics[
                (metrics.source == "Trained")
                & (metrics.dataset == dataset)
                & (metrics.corpus == corpus)
            ]
            vals.append(100 * float(sub.acc.mean()))
        offsets = [pos + (offset_idx - 1.5) * width for pos in x]
        ax_corpus.bar(
            offsets,
            vals,
            width=width,
            label=CORPUS_LABELS[corpus],
            color=corpus_colors[corpus],
            edgecolor="black",
            linewidth=0.4,
        )
    ax_corpus.set_title("(b) Ours by retrieval corpus", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_corpus.set_ylabel("Mean accuracy across models (%)")
    ax_corpus.set_xticks(list(x), [DATASET_LABELS[d] for d in dataset_keys])
    ax_corpus.set_ylim(20, 80)
    ax_corpus.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax_corpus.legend(ncols=2, frameon=False, loc="upper left")

    dataset_colors = {
        "medqa_usmle": NATURE_COLORS["blue"],
        "headqa": NATURE_COLORS["green"],
        "pubmedqa_labeled": NATURE_COLORS["vermillion"],
    }
    markers = {"pubmed": "o", "textbooks": "s", "statpearls": "^", "wikipedia": "D"}
    matched_rows: list[tuple[str, str, str, float, float]] = []
    for dataset in dataset_keys:
        for model in model_keys:
            for corpus in corpus_keys:
                search = metrics[
                    (metrics.source == "Baseline")
                    & (metrics.dataset == dataset)
                    & (metrics.model == model)
                    & (metrics.experiment == "Search prompt")
                    & (metrics.corpus == corpus)
                ]
                ours = metrics[
                    (metrics.source == "Trained")
                    & (metrics.dataset == dataset)
                    & (metrics.model == model)
                    & (metrics.corpus == corpus)
                ]
                if not search.empty and not ours.empty:
                    matched_rows.append((dataset, model, corpus, 100 * float(search.iloc[0].acc), 100 * float(ours.iloc[0].acc)))

    for dataset, model, corpus, search_acc, ours_acc in matched_rows:
        ax_pair.scatter(
            search_acc,
            ours_acc,
            s=42,
            marker=markers[corpus],
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.4,
            alpha=0.86,
        )
    lo = min(min(row[3], row[4]) for row in matched_rows) - 2
    hi = max(max(row[3], row[4]) for row in matched_rows) + 2
    ax_pair.plot([lo, hi], [lo, hi], color="#666666", linestyle="--", linewidth=1)
    ax_pair.set_xlim(lo, hi)
    ax_pair.set_ylim(lo, hi)
    ax_pair.set_title("(c) Search prompt vs trained search", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_pair.set_xlabel("Search prompt accuracy (%)")
    ax_pair.set_ylabel("Ours accuracy (%)")
    ax_pair.grid(color="#d9d9d9", linewidth=0.7, alpha=0.7)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=dataset_colors[d], markeredgecolor="black", label=DATASET_LABELS[d])
        for d in dataset_keys
    ]
    corpus_handles = [
        plt.Line2D([0], [0], marker=markers[c], color="black", linestyle="none", markerfacecolor="white", label=CORPUS_LABELS[c])
        for c in corpus_keys
    ]
    leg1 = ax_pair.legend(handles=dataset_handles, frameon=False, loc="upper left", title="Dataset")
    ax_pair.add_artist(leg1)
    ax_pair.legend(handles=corpus_handles, frameon=False, loc="lower right", title="Corpus")

    best_rows: list[tuple[str, str, float, float]] = []
    for dataset in dataset_keys:
        for model in model_keys:
            sub = metrics[(metrics.dataset == dataset) & (metrics.model == model)]
            best_baseline = float(sub[sub.source == "Baseline"].acc.max()) * 100
            best_ours = float(sub[sub.source == "Trained"].acc.max()) * 100
            label = f"{MODEL_SHORT[model]} / {DATASET_LABELS[dataset]}"
            best_rows.append((label, dataset, best_baseline, best_ours))
    y_pos = list(range(len(best_rows)))
    for y, (label, dataset, baseline_acc, ours_acc) in zip(y_pos, best_rows):
        ax_best.plot(
            [baseline_acc, ours_acc],
            [y, y],
            color="#9E9E9E",
            linewidth=1.3,
            zorder=1,
        )
        ax_best.scatter(
            baseline_acc,
            y,
            s=42,
            color="white",
            edgecolor=NATURE_COLORS["gray"],
            linewidth=1.0,
            zorder=2,
            label="Best untrained" if y == 0 else None,
        )
        ax_best.scatter(
            ours_acc,
            y,
            s=48,
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
            label="Best Ours" if y == 0 else None,
        )
        ax_best.text(
            max(baseline_acc, ours_acc) + 0.7,
            y,
            f"+{ours_acc - baseline_acc:.1f}",
            va="center",
            fontsize=9,
        )
    ax_best.set_title("(d) Best baseline vs best trained search", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_best.set_yticks(y_pos, [row[0] for row in best_rows])
    ax_best.invert_yaxis()
    ax_best.set_xlabel("Accuracy (%)")
    ax_best.set_xlim(10, 85)
    ax_best.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax_best.legend(frameon=False, loc="upper right")

    best_corpus_counts = {corpus: 0 for corpus in corpus_keys}
    best_corpus_by_dataset = {dataset: {corpus: 0 for corpus in corpus_keys} for dataset in dataset_keys}
    for dataset in dataset_keys:
        for model in model_keys:
            trained = metrics[
                (metrics.source == "Trained")
                & (metrics.dataset == dataset)
                & (metrics.model == model)
            ]
            if trained.empty:
                continue
            corpus = str(trained.loc[trained.acc.idxmax()].corpus)
            best_corpus_counts[corpus] += 1
            best_corpus_by_dataset[dataset][corpus] += 1
    bottom = [0] * len(corpus_keys)
    x_choice = list(range(len(corpus_keys)))
    for dataset in dataset_keys:
        vals = [best_corpus_by_dataset[dataset][corpus] for corpus in corpus_keys]
        ax_choice.bar(
            x_choice,
            vals,
            bottom=bottom,
            color=dataset_colors[dataset],
            edgecolor="black",
            linewidth=0.4,
            label=DATASET_LABELS[dataset],
        )
        bottom = [old + val for old, val in zip(bottom, vals)]
    for xpos, corpus in zip(x_choice, corpus_keys):
        total = best_corpus_counts[corpus]
        if total:
            ax_choice.text(xpos, total + 0.08, str(total), ha="center", va="bottom", fontsize=10)
    ax_choice.set_title("(e) Corpus selected by best Ours", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_choice.set_xticks(x_choice, [CORPUS_LABELS[c] for c in corpus_keys], rotation=20, ha="right")
    ax_choice.set_ylabel("Number of model-dataset pairs")
    ax_choice.set_ylim(0, max(best_corpus_counts.values()) + 1.1)
    ax_choice.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax_choice.legend(frameon=False, ncols=3, loc="upper right")

    delta_values: dict[str, list[float]] = {corpus: [] for corpus in corpus_keys}
    for dataset, model, corpus, search_acc, ours_acc in matched_rows:
        delta_values[corpus].append(ours_acc - search_acc)
    delta_means = [sum(delta_values[corpus]) / len(delta_values[corpus]) for corpus in corpus_keys]
    delta_err = []
    for corpus in corpus_keys:
        vals = delta_values[corpus]
        mean = sum(vals) / len(vals)
        variance = sum((val - mean) ** 2 for val in vals) / len(vals)
        delta_err.append(math.sqrt(variance))
    ax_delta.bar(
        x_choice,
        delta_means,
        yerr=delta_err,
        capsize=3,
        color=[corpus_colors[corpus] for corpus in corpus_keys],
        edgecolor="black",
        linewidth=0.4,
    )
    for xpos, value in zip(x_choice, delta_means):
        ax_delta.text(xpos, value + 0.8, f"{value:+.1f}", ha="center", va="bottom", fontsize=10)
    ax_delta.axhline(0, color=NATURE_COLORS["gray"], linewidth=0.8)
    ax_delta.set_title("(f) Training gain over Search prompt", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE, pad=PANEL_TITLE_PAD)
    ax_delta.set_xticks(x_choice, [CORPUS_LABELS[c] for c in corpus_keys], rotation=20, ha="right")
    ax_delta.set_ylabel("Mean accuracy difference (pp)")
    ax_delta.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)

    fig.subplots_adjust(left=0.115, right=0.99, top=0.975, bottom=0.05, wspace=0.30, hspace=0.34)
    fig.savefig(FIG / "fig_main_results_summary.pdf")
    fig.savefig(FIG / "fig_main_results_summary.png", dpi=300)
    plt.close(fig)


def csv_exports(main: pd.DataFrame, base: pd.DataFrame, abl: pd.DataFrame) -> None:
    main.to_csv(TAB / "main_grpo.csv", index=False)
    base.to_csv(TAB / "baselines_all_with_fewshot.csv", index=False)
    abl.to_csv(TAB / "ablation_pubmedqa.csv", index=False)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    main_df, base_df, abl_df = load()
    write_main_experiment_table(main_df, base_df)
    metrics_df = pd.read_csv(TAB / "main_metrics.csv")
    write_main_result_figure(metrics_df)
    write_ablation_figure(abl_df)
    csv_exports(main_df, base_df, abl_df)


if __name__ == "__main__":
    main()
