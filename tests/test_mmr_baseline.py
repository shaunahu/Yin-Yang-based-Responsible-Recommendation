import pytest

from model.mmr_baseline import cosine_similarity, rerank_with_mmr


def test_cosine_similarity_returns_zero_for_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_mmr_keeps_highest_relevance_first_when_lambda_is_high():
    reranked = rerank_with_mmr(
        user_id="U1",
        candidate_items=["A", "B", "C"],
        user_embeddings={"1": [1.0, 0.0]},
        item_embeddings={
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
        },
        user_token_map={"U1": "1"},
        item_token_map={"A": "a", "B": "b", "C": "c"},
        lambda_weight=0.9,
        top_k=3,
    )

    assert reranked == ["A", "B", "C"]


def test_mmr_promotes_diversity_when_lambda_is_low():
    reranked = rerank_with_mmr(
        user_id="U1",
        candidate_items=["A", "B", "C"],
        user_embeddings={"1": [1.0, 0.0]},
        item_embeddings={
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
        },
        user_token_map={"U1": "1"},
        item_token_map={"A": "a", "B": "b", "C": "c"},
        lambda_weight=0.2,
        top_k=3,
    )

    assert reranked == ["A", "C", "B"]


def test_mmr_limits_output_to_top_k_and_deduplicates_candidates():
    reranked = rerank_with_mmr(
        user_id="U1",
        candidate_items=["A", "A", "B"],
        user_embeddings={},
        item_embeddings={},
        user_token_map={},
        item_token_map={},
        lambda_weight=0.7,
        top_k=2,
    )

    assert reranked == ["A", "B"]


def test_mmr_rejects_invalid_lambda():
    with pytest.raises(ValueError):
        rerank_with_mmr("U1", ["A"], {}, {}, {}, {}, lambda_weight=1.1, top_k=1)
