from collections import OrderedDict

from model.evaluate_recommendations import (
    evaluate_recommendations,
    format_ordered_dict,
    parse_positive_items,
)


def test_parse_positive_items_ignores_negative_impressions():
    assert parse_positive_items("A-1 B-0 C-1 malformed") == {"A", "C"}


def test_evaluate_recommendations_computes_ranking_metrics():
    recommendations = [
        {"user_id": "U1", "recommended_items": ["A", "B", "C"]},
        {"user_id": "U2", "recommended_items": ["D", "E", "F"]},
    ]
    ground_truth = {
        "U1": {"B", "C"},
        "U2": {"X"},
    }

    metrics = evaluate_recommendations(
        recommendations,
        ground_truth,
        topk_values=[2],
    )

    assert metrics == OrderedDict(
        [
            ("recall@2", 0.25),
            ("ndcg@2", 0.1934),
            ("hit@2", 0.5),
            ("precision@2", 0.25),
            ("mrr@2", 0.25),
            ("evaluated_users", 2),
        ]
    )


def test_format_ordered_dict_matches_result_log_style():
    metrics = OrderedDict([("recall@5", 0.1), ("ndcg@5", 0.2)])

    assert format_ordered_dict(metrics) == "OrderedDict([('recall@5', 0.1), ('ndcg@5', 0.2)])"
