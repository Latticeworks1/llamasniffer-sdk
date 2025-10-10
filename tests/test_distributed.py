#!/usr/bin/env python3
"""
Test script for distributed Ollama inference capabilities.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llamasniffer


def test_basic_discovery():
    """Test basic local discovery functionality."""
    print("=== Testing Local Discovery ===")
    instances = llamasniffer.discover_ollama_instances()
    print(f"Found {len(instances)} local instances")

    if instances:
        print("\nInstance details:")
        for i, instance in enumerate(instances):
            print(
                f"  {i + 1}. {instance['host']}:{instance['port']} - {len(instance.get('models', []))} models"
            )
            print(f"     Response time: {instance.get('version_response_time_ms', 'unknown')}ms")

    return instances


def test_distributed_manager(instances):
    """Test distributed manager functionality."""
    if not instances:
        print("No instances available for distributed testing")
        return

    print("\n=== Testing Distributed Manager ===")

    # Create manager with fastest strategy
    manager = llamasniffer.create_distributed_manager(instances=instances, strategy="fastest")

    # Get cluster status
    status = manager.get_cluster_status()
    print(f"\nCluster Status:")
    print(f"  Health: {status['cluster_health']['health_percentage']}%")
    print(
        f"  Instances: {status['cluster_health']['healthy_instances']}/{status['cluster_health']['total_instances']}"
    )
    print(f"  Available models: {status['model_availability']['unique_models']}")
    print(f"  Models: {status['model_availability']['models'][:3]}...")  # Show first 3 models

    # Test inference if models available
    available_models = status["model_availability"]["models"]
    if available_models:
        test_model = available_models[0]
        print(f"\n=== Testing Inference with {test_model} ===")

        # Single inference
        response = manager.generate_distributed(
            test_model, "What is 2 + 2? Answer briefly.", max_retries=1
        )

        if "error" not in response:
            print("✓ Single inference successful")
            print(f"  Instance: {response['execution_metadata']['instance']}")
            print(f"  Response time: {response['execution_metadata']['response_time_ms']}ms")
            print(f"  Response: {response.get('response', 'No response')[:100]}...")
        else:
            print(f"✗ Single inference failed: {response['error']}")

        # Parallel inference if multiple instances
        if len(instances) > 1:
            print("\n=== Testing Parallel Inference ===")
            parallel_response = manager.generate_distributed(
                test_model,
                "What is the capital of Japan? Answer briefly.",
                parallel_requests=min(2, len(instances)),
            )

            if "error" not in parallel_response:
                print(f"✓ Parallel inference successful")
                print(f"  Best instance: {parallel_response['execution_metadata']['instance']}")
                print(f"  Parallel results: {len(parallel_response.get('parallel_results', []))}")
            else:
                print(f"✗ Parallel inference failed: {parallel_response['error']}")
    else:
        print("No models available for inference testing")


def test_load_balancing_strategies(instances):
    """Test different load balancing strategies."""
    if len(instances) < 2:
        print("Need at least 2 instances to test load balancing")
        return

    print("\n=== Testing Load Balancing Strategies ===")

    strategies = ["fastest", "round_robin", "least_loaded"]

    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        manager = llamasniffer.create_distributed_manager(instances=instances, strategy=strategy)

        # Get first available model
        status = manager.get_cluster_status()
        if status["model_availability"]["models"]:
            model = status["model_availability"]["models"][0]

            # Run 3 quick inferences to see distribution
            for i in range(3):
                response = manager.generate_distributed(
                    model, f"Test query {i + 1} for {strategy}", max_retries=0
                )
                if "error" not in response:
                    instance = response["execution_metadata"]["instance"]
                    response_time = response["execution_metadata"]["response_time_ms"]
                    print(f"  Query {i + 1}: {instance} ({response_time}ms)")


if __name__ == "__main__":
    print("LlamaSniffer - Distributed Inference Test")
    print("=" * 50)

    # Test discovery
    instances = test_basic_discovery()

    # Test distributed functionality
    if instances:
        test_distributed_manager(instances)
        test_load_balancing_strategies(instances)
    else:
        print("\nNo instances found. Make sure Ollama is running on your network.")
        print("You can also test with Shodan discovery if you have an API key.")

    print("\n" + "=" * 50)
    print("Test completed")
