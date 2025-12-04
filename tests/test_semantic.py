import numpy as np

from llamasniffer.core import SemanticModelMatcher


def _make_keyword_embedding(text: str) -> np.ndarray:
    text = text.lower()
    vec = np.zeros(3)
    if "reasoning" in text or "deepseek" in text:
        vec[0] = 1.0
    if "coding" in text:
        vec[1] = 1.0
    if "creative" in text:
        vec[2] = 1.0
    if not vec.any():
        vec[-1] = 0.1
    return vec


def test_find_best_model_prefers_semantic_match(monkeypatch):
    matcher = SemanticModelMatcher()
    monkeypatch.setattr(matcher, "_get_embedding", _make_keyword_embedding)

    models = ["deepseek-coder:6.7b", "gemma:2b-instruct"]
    result = matcher.find_best_model("reasoning task", models, similarity_threshold=0.1)

    assert result is not None
    assert result["model"] == "deepseek-coder:6.7b"
    assert result["method"] in {"semantic_description", "direct_semantic"}


def test_explain_model_choice_without_embeddings(monkeypatch):
    matcher = SemanticModelMatcher()
    monkeypatch.setattr(matcher, "_get_embedding", lambda text: None)

    explanation = matcher.explain_model_choice("unknown", [])

    assert explanation["embedding_available"] is False
    assert "error" in explanation
