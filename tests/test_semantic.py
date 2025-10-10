#!/usr/bin/env python3
"""
Test script for semantic model matching functionality.
Tests the integration with latterworks/ollama-embeddings model.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llamasniffer


def test_embedding_connection():
    """Test connection to local embedding model."""
    print("=== Testing Embedding Model Connection ===")

    matcher = llamasniffer.SemanticModelMatcher()
    test_embedding = matcher._get_embedding("test query")

    if test_embedding is not None:
        print("✓ Successfully connected to embedding model")
        print(f"  Embedding dimension: {test_embedding.shape[0]}")
        return True
    else:
        print("✗ Could not connect to embedding model")
        print("  Make sure LM Studio is running with text-embedding-nomic-embed-text-v1.5")
        return False


def test_semantic_matching():
    """Test semantic model matching with mock models."""
    print("\n=== Testing Semantic Model Matching ===")

    # Mock available models (typical Ollama model names)
    mock_models = [
        "llama2:7b-chat",
        "codellama:13b-instruct",
        "mistral:7b-instruct",
        "deepseek-coder:6.7b",
        "phi3:3.8b",
        "qwen:7b-chat",
        "gemma:2b-instruct",
    ]

    matcher = llamasniffer.SemanticModelMatcher()

    # Test queries
    test_queries = [
        "reasoning",
        "coding",
        "creative writing",
        "fast model",
        "small efficient model",
        "mathematical problem solving",
        "programming assistance",
    ]

    print(f"Available models: {mock_models}")
    print("\nTesting semantic queries:")

    for query in test_queries:
        result = matcher.find_best_model(query, mock_models)

        if result:
            print(f"  '{query}' → {result['model']} (confidence: {result['confidence']:.3f})")
            print(f"    Method: {result['method']}")
            if "matched_concept" in result:
                print(f"    Concept: {result['matched_concept']}")
        else:
            print(f"  '{query}' → No match found")

    # Test explanation feature
    print("\n=== Detailed Explanation for 'reasoning' ===")
    explanation = matcher.explain_model_choice("reasoning", mock_models)
    print(json.dumps(explanation, indent=2))


def test_distributed_semantic():
    """Test distributed manager with semantic matching."""
    print("\n=== Testing Distributed Semantic Inference ===")

    # Discover real instances
    instances = llamasniffer.discover_ollama_instances()

    if not instances:
        print("No local instances found for testing")
        return

    print(f"Found {len(instances)} instances")

    # Create manager with semantic matching enabled
    manager = llamasniffer.create_distributed_manager(instances=instances, strategy="fastest")

    # Get cluster status
    status = manager.get_cluster_status()
    available_models = status["model_availability"]["models"]

    if not available_models:
        print("No models available for semantic testing")
        return

    print(f"Available models: {available_models[:3]}...")  # Show first 3

    # Test semantic queries
    semantic_queries = ["reasoning", "coding", "fast", "creative"]

    for query in semantic_queries:
        print(f"\nTesting query: '{query}'")

        # Test model resolution only
        resolution = manager._resolve_model_name(query)
        if resolution:
            print(f"  Resolved to: {resolution['model']}")
            print(f"  Method: {resolution['method']}")
            if "confidence" in resolution:
                print(f"  Confidence: {resolution['confidence']:.3f}")

        # Test full inference (with short prompt)
        try:
            response = manager.generate_distributed(
                query, "What is 2 + 2? Answer briefly.", max_retries=1
            )

            if "error" not in response:
                print("  ✓ Inference successful")
                model_res = response["execution_metadata"]["model_resolution"]
                print(f"    Used model: {model_res['model']}")
                print(f"    Resolution method: {model_res['method']}")
            else:
                print(f"  ✗ Inference failed: {response['error']}")

        except Exception as e:
            print(f"  ✗ Exception during inference: {e}")


def main():
    print("LlamaSniffer - Semantic Model Matching Test")
    print("=" * 55)

    # Test embedding connection first
    embedding_works = test_embedding_connection()

    # Test semantic matching (works even without embeddings)
    test_semantic_matching()

    # Test distributed semantic inference
    if embedding_works:
        test_distributed_semantic()
    else:
        print("\nSkipping distributed tests - embedding model not available")
        print("Start LM Studio with text-embedding-nomic-embed-text-v1.5 to enable full testing")

    print("\n" + "=" * 55)
    print("Semantic testing completed")


if __name__ == "__main__":
    main()
