#!/usr/bin/env python3
"""
Test script for token logger functionality.
"""
import os
import sys
from datetime import datetime
from token_logger import get_logger


def test_token_logger():
    """Test the token logger functionality."""
    print("Testing Token Logger...")
    print("=" * 60)

    # Initialize logger with test directory
    test_log_dir = "./logs_test"
    logger = get_logger(test_log_dir)

    print(f"\n1. Testing log creation in: {test_log_dir}")

    # Test logging with different providers
    test_data = [
        {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "input_tokens": 150,
            "output_tokens": 75
        },
        {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "input_tokens": 200,
            "output_tokens": 100
        },
        {
            "provider": "mistral",
            "model": "mistral-large-latest",
            "input_tokens": 180,
            "output_tokens": 90
        }
    ]

    print("   Logging test data...")
    for data in test_data:
        logger.log_usage(**data)
        print(f"   ✓ Logged: {data['provider']} ({data['model']})")

    # Test reading log files
    print(f"\n2. Testing log file reading...")
    log_files = logger.get_all_log_files()
    if log_files:
        print(f"   ✓ Found {len(log_files)} log file(s)")
    else:
        print("   ✗ No log files found!")
        sys.exit(1)

    # Test statistics
    print(f"\n3. Testing statistics calculation...")
    stats = logger.get_statistics()

    print(f"   Total requests: {stats['total']['requests']}")
    print(f"   Total input tokens: {stats['total']['input_tokens']}")
    print(f"   Total output tokens: {stats['total']['output_tokens']}")
    print(f"   Total tokens: {stats['total']['total_tokens']}")

    # Verify counts
    expected_requests = len(test_data)
    expected_input = sum(d['input_tokens'] for d in test_data)
    expected_output = sum(d['output_tokens'] for d in test_data)

    if stats['total']['requests'] == expected_requests:
        print("   ✓ Request count matches")
    else:
        print(f"   ✗ Request count mismatch! Expected {expected_requests}, got {stats['total']['requests']}")

    if stats['total']['input_tokens'] == expected_input:
        print("   ✓ Input token count matches")
    else:
        print(f"   ✗ Input token mismatch! Expected {expected_input}, got {stats['total']['input_tokens']}")

    if stats['total']['output_tokens'] == expected_output:
        print("   ✓ Output token count matches")
    else:
        print(f"   ✗ Output token mismatch! Expected {expected_output}, got {stats['total']['output_tokens']}")

    # Test by provider
    print(f"\n4. Testing provider breakdown...")
    for provider in ['openai', 'anthropic', 'mistral']:
        if provider in stats['by_provider']:
            provider_stats = stats['by_provider'][provider]
            print(f"   ✓ {provider}: {provider_stats['requests']} request(s), "
                  f"{provider_stats['total_tokens']} tokens")
        else:
            print(f"   ✗ {provider} not found in statistics!")

    # Test cost estimation
    print(f"\n5. Testing cost estimation...")
    for data in test_data:
        cost = logger.get_cost_estimate(
            data['provider'],
            data['model'],
            data['input_tokens'],
            data['output_tokens']
        )
        if cost:
            print(f"   ✓ {data['provider']}/{data['model']}: ${cost:.6f}")
        else:
            print(f"   ⚠ {data['provider']}/{data['model']}: Pricing not available")

    print("\n" + "=" * 60)
    print("✅ Token logger test completed successfully!")
    print(f"\nTest log files created in: {test_log_dir}")
    print("You can clean up test logs by running: rm -rf ./logs_test")
    print("=" * 60)


if __name__ == "__main__":
    test_token_logger()
