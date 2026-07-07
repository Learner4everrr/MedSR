import os
import re
import string


def _normalize(text):
    text = str(text).strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return "".join(text.split())


def _targets(ground_truth):
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("answer", ground_truth))
    if isinstance(ground_truth, str):
        return [ground_truth]
    try:
        return [str(x) for x in list(ground_truth)]
    except TypeError:
        return [str(ground_truth)]


def _extract_answer(text):
    matches = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", str(text), flags=re.DOTALL | re.I))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _tag_balanced(text, tag):
    return len(re.findall(fr"<{tag}>", text, flags=re.I)) == len(re.findall(fr"</{tag}>", text, flags=re.I))


def _format_ok(text, answer_valid):
    if not answer_valid:
        return False
    text = str(text)
    if not re.search(r"<search>.*?</search>", text, flags=re.I | re.DOTALL):
        return False
    if not re.search(r"<information>.*?</information>", text, flags=re.I | re.DOTALL):
        return False
    if not _tag_balanced(text, "think"):
        return False
    if not _tag_balanced(text, "search"):
        return False
    if not _tag_balanced(text, "information"):
        return False
    return True


def _has_search(text):
    return re.search(r"<search>.*?</search>", str(text), flags=re.I | re.DOTALL) is not None


def _length_penalty(text):
    max_chars = int(os.getenv("BIOMED_LENGTH_PENALTY_MAX_CHARS", "2500"))
    penalty = float(os.getenv("BIOMED_LENGTH_PENALTY", "0.1"))
    return penalty if len(str(text)) > max_chars else 0.0


def _combine_reward(solution_str, correct, format_ok):
    mode = os.getenv("BIOMED_REWARD_MODE", "default").strip().lower()
    format_reward = float(os.getenv("BIOMED_FORMAT_REWARD", "0.1"))
    if mode in {"default", "answer_plus_format"}:
        return correct * (1.0 - format_reward) + format_reward * format_ok
    if mode == "accuracy_only":
        return correct
    if mode == "format_only":
        return format_ok
    if mode == "answer_plus_format_heavy":
        heavy_format_reward = float(os.getenv("BIOMED_HEAVY_FORMAT_REWARD", "0.3"))
        return correct * (1.0 - heavy_format_reward) + heavy_format_reward * format_ok
    if mode == "answer_plus_search_light":
        search_reward = float(os.getenv("BIOMED_SEARCH_REWARD", "0.05"))
        base = correct * (1.0 - format_reward - search_reward)
        return base + format_reward * format_ok + search_reward * float(_has_search(solution_str))
    if mode == "answer_plus_length_penalty":
        return max(0.0, correct * (1.0 - format_reward) + format_reward * format_ok - _length_penalty(solution_str))
    raise ValueError(f"Unknown BIOMED_REWARD_MODE={mode}")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    answer = _extract_answer(solution_str)
    if answer is None:
        return {"score": 0.0, "format": 0.0, "accuracy": 0.0, "answer": ""}

    dataset = str(data_source).lower()
    if "pubmedqa" in dataset:
        valid = re.fullmatch(r"\s*(yes|no|maybe)\s*", answer, flags=re.I) is not None
    else:
        valid = re.fullmatch(r"\s*[A-Za-z]\s*", answer) is not None

    if not valid:
        return {"score": 0.0, "format": 0.0, "accuracy": 0.0, "answer": answer}

    pred = _normalize(answer)
    gold = [_normalize(x) for x in _targets(ground_truth)]
    correct = float(pred in gold)
    format_ok = float(_format_ok(solution_str, valid))
    score = _combine_reward(solution_str, correct, format_ok)
    return {"score": score, "format": format_ok, "accuracy": correct, "answer": answer}
