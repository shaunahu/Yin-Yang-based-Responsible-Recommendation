"""
Evaluate saved recommendation pickle files.

The evaluator compares generated recommendation lists against the behavior
test-set positives.
"""

import argparse
import math
from collections import OrderedDict
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from model.mmr_baseline import load_pickle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> ConfigParser:
    config = ConfigParser()
    config.read(config_path)
    return config


def parse_positive_items(impression: str) -> Set[str]:
    positives = set()
    for token in impression.split():
        try:
            item_id, label = token.rsplit("-", 1)
        except ValueError:
            continue
        if label == "1":
            positives.add(item_id)
    return positives


def load_ground_truth(behavior_path: Path) -> Dict[str, Set[str]]:
    with behavior_path.open() as f:
        header = f.readline().strip().split("\t")
        try:
            user_idx = header.index("userid")
            impression_idx = header.index("impression")
        except ValueError as exc:
            raise ValueError(
                f"{behavior_path} must contain userid and impression columns"
            ) from exc

        ground_truth = {}
        for line in f:
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= max(user_idx, impression_idx):
                continue
            positives = parse_positive_items(columns[impression_idx])
            if positives:
                ground_truth[columns[user_idx]] = positives
        return ground_truth


def reciprocal_rank(
    recommended_items: Sequence[str], relevant_items: Set[str], k: int
) -> float:
    for rank, item_id in enumerate(recommended_items[:k], start=1):
        if item_id in relevant_items:
            return 1.0 / rank
    return 0.0


def dcg_at_k(recommended_items: Sequence[str], relevant_items: Set[str], k: int) -> float:
    return sum(
        1.0 / log2(rank + 1)
        for rank, item_id in enumerate(recommended_items[:k], start=1)
        if item_id in relevant_items
    )


def log2(value: int) -> float:
    if hasattr(value, "bit_length"):
        return math.log2(value)
    return math.log(value, 2)


def ndcg_at_k(recommended_items: Sequence[str], relevant_items: Set[str], k: int) -> float:
    ideal_hits = min(len(relevant_items), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(recommended_items, relevant_items, k) / ideal_dcg


def evaluate_recommendations(
    recommendations: Iterable[Mapping[str, object]],
    ground_truth: Mapping[str, Set[str]],
    topk_values: Sequence[int],
) -> OrderedDict:
    totals = OrderedDict()
    for k in topk_values:
        totals[f"recall@{k}"] = 0.0
        totals[f"ndcg@{k}"] = 0.0
        totals[f"hit@{k}"] = 0.0
        totals[f"precision@{k}"] = 0.0
        totals[f"mrr@{k}"] = 0.0

    evaluated_users = 0
    for recommendation in recommendations:
        user_id = str(recommendation["user_id"])
        relevant_items = ground_truth.get(user_id)
        if not relevant_items:
            continue

        evaluated_users += 1
        recommended_items = [
            str(item_id) for item_id in recommendation["recommended_items"]
        ]

        for k in topk_values:
            hits = len(set(recommended_items[:k]) & relevant_items)
            totals[f"recall@{k}"] += hits / len(relevant_items)
            totals[f"ndcg@{k}"] += ndcg_at_k(recommended_items, relevant_items, k)
            totals[f"hit@{k}"] += 1.0 if hits else 0.0
            totals[f"precision@{k}"] += hits / k
            totals[f"mrr@{k}"] += reciprocal_rank(recommended_items, relevant_items, k)

    if evaluated_users == 0:
        raise ValueError("No recommendation users matched the behavior ground truth")

    metrics = OrderedDict()
    for metric_name, total in totals.items():
        metrics[metric_name] = round(total / evaluated_users, 4)

    metrics["evaluated_users"] = evaluated_users
    return metrics


def resolve_save_dir(args) -> Path:
    if args.save_dir:
        return Path(args.save_dir)

    config = load_config(PROJECT_ROOT / "common" / "parameter.ini")
    dataset = args.dataset or config.get("simulation", "dataset")
    recommender = args.recommender or config.get("simulation", "recommender")
    return PROJECT_ROOT / "saved" / dataset / recommender


def infer_dataset(save_dir: Path, args_dataset: Optional[str]) -> str:
    if args_dataset:
        return args_dataset
    try:
        return save_dir.relative_to(PROJECT_ROOT / "saved").parts[0]
    except (ValueError, IndexError):
        config = load_config(PROJECT_ROOT / "common" / "parameter.ini")
        return config.get("simulation", "dataset")


def append_result_log(
    log_path: Path,
    model_name: str,
    recommendation_file: str,
    metrics: OrderedDict,
    behavior_path: Path,
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Recommendation file: {recommendation_file}\n")
        f.write(f"Behavior file: {behavior_path}\n")
        f.write(f"Time: {now}\n")
        f.write(f"Test result: {format_ordered_dict(metrics)}\n")
        f.write("=" * 50 + "\n")


def format_ordered_dict(metrics: OrderedDict) -> str:
    return f"OrderedDict({list(metrics.items())})"


def parse_topk(topk: str) -> List[int]:
    return [int(value.strip()) for value in topk.split(",") if value.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate recommendation pickle files and append a results log."
    )
    parser.add_argument("--dataset", help="Dataset name under resource/ and saved/.")
    parser.add_argument("--recommender", help="Recommender folder under saved/<dataset>/.")
    parser.add_argument("--save-dir", help="Explicit directory containing recommendation files.")
    parser.add_argument(
        "--files",
        nargs="+",
        default=["recommendations.pkl", "mmr_recommendations.pkl"],
        help="Recommendation pickle files inside save-dir.",
    )
    parser.add_argument("--topk", default="5,10,20")
    parser.add_argument("--log-name", default="results.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = resolve_save_dir(args)
    dataset = infer_dataset(save_dir, args.dataset)
    behavior_path = PROJECT_ROOT / "resource" / dataset / "new_behaviors.tsv"
    ground_truth = load_ground_truth(behavior_path)
    topk_values = parse_topk(args.topk)

    log_path = save_dir / args.log_name
    for recommendation_file in args.files:
        recommendation_path = save_dir / recommendation_file
        if not recommendation_path.exists():
            print(f"Skip missing file: {recommendation_path}")
            continue

        recommendations = load_pickle(recommendation_path)
        metrics = evaluate_recommendations(
            recommendations,
            ground_truth,
            topk_values,
        )
        append_result_log(
            log_path,
            recommendation_path.stem,
            recommendation_file,
            metrics,
            behavior_path,
        )
        print(f"{recommendation_path.stem}: {format_ordered_dict(metrics)}")

    print(f"Appended evaluation results to {log_path}")


if __name__ == "__main__":
    main()
