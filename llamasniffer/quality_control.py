"""
Quality control for discovered Ollama endpoints.

Tests endpoints with validation prompts to filter out broken/garbled models.
"""

import re
import time
import requests
from typing import Dict, List, Optional, Tuple


class EndpointQualityChecker:
    """Quality validation for Ollama endpoints.

    Tests endpoints with simple prompts and validates responses
    to filter out broken models that produce garbled output.
    """

    # Simple validation prompts that should produce predictable responses
    VALIDATION_PROMPTS = [
        {
            'prompt': 'What is 2+2? Answer with only the number.',
            'expected_keywords': ['4', 'four'],
            'max_length': 50,
            'description': 'basic_math'
        },
        {
            'prompt': 'Complete this: The sky is ___. Answer with one word.',
            'expected_keywords': ['blue', 'gray', 'grey', 'clear', 'cloudy'],
            'max_length': 30,
            'description': 'simple_completion'
        }
    ]

    def __init__(self,
                 timeout: float = 10.0,
                 min_quality_score: float = 0.5):
        """Initialize quality checker.

        Args:
            timeout: Timeout for quality test requests
            min_quality_score: Minimum score to pass (0.0-1.0)
        """
        self.timeout = timeout
        self.min_quality_score = min_quality_score

    def check_endpoint_quality(self, endpoint: Dict) -> Tuple[float, Dict]:
        """Test endpoint quality with validation prompts.

        Args:
            endpoint: Endpoint dict with 'host', 'port', 'models'

        Returns:
            Tuple of (quality_score, test_results)
        """
        url = f"http://{endpoint['host']}:{endpoint['port']}"
        models = endpoint.get('models', [])

        if not models:
            return 0.0, {'error': 'no_models'}

        # Test first available model
        model = models[0]

        total_score = 0.0
        test_results = {
            'model_tested': model,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': []
        }

        for validation in self.VALIDATION_PROMPTS:
            score, details = self._test_single_prompt(
                url, model, validation
            )

            total_score += score
            test_results['test_details'].append(details)

            if score >= 0.5:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1

        # Calculate average score
        avg_score = total_score / len(self.VALIDATION_PROMPTS)
        test_results['quality_score'] = avg_score

        return avg_score, test_results

    def _test_single_prompt(self,
                           url: str,
                           model: str,
                           validation: Dict) -> Tuple[float, Dict]:
        """Test endpoint with a single validation prompt.

        Returns:
            Tuple of (score, details)
        """
        details = {
            'test': validation['description'],
            'prompt': validation['prompt'],
            'success': False,
            'score': 0.0
        }

        try:
            # Send generation request
            start = time.time()
            response = requests.post(
                f"{url}/api/generate",
                json={
                    'model': model,
                    'prompt': validation['prompt'],
                    'stream': False
                },
                timeout=self.timeout
            )
            response_time = (time.time() - start) * 1000

            if response.status_code != 200:
                details['error'] = f"HTTP {response.status_code}"
                return 0.0, details

            data = response.json()
            generated_text = data.get('response', '').strip()

            details['response'] = generated_text[:100]
            details['response_time_ms'] = round(response_time, 2)

            # Validate response quality
            score = self._score_response(
                generated_text,
                validation['expected_keywords'],
                validation['max_length']
            )

            details['score'] = score
            details['success'] = score >= 0.5

            return score, details

        except requests.Timeout:
            details['error'] = 'timeout'
            return 0.0, details
        except Exception as e:
            details['error'] = str(e)[:100]
            return 0.0, details

    def _score_response(self,
                       response: str,
                       expected_keywords: List[str],
                       max_length: int) -> float:
        """Score a response based on quality criteria.

        Returns:
            Score from 0.0 (garbage) to 1.0 (perfect)
        """
        if not response:
            return 0.0

        response_lower = response.lower()
        score = 0.0

        # Check 1: Contains expected keywords (0.5 points)
        keyword_found = any(kw in response_lower for kw in expected_keywords)
        if keyword_found:
            score += 0.5

        # Check 2: Reasonable length (0.2 points)
        if len(response) <= max_length:
            score += 0.2

        # Check 3: Not garbled (0.3 points)
        # Garbled text has lots of random characters and numbers
        if self._is_coherent(response):
            score += 0.3

        return min(1.0, score)

    def _is_coherent(self, text: str) -> bool:
        """Check if text is coherent (not garbled).

        Returns:
            True if text appears coherent
        """
        if not text:
            return False

        # Count random-looking patterns
        random_score = 0

        # Too many numbers mixed with letters (like "8eq cw9ov r96qrcf")
        digit_letter_mix = len(re.findall(r'\d+[a-z]+|\d+', text, re.I))
        if digit_letter_mix > len(text.split()) * 0.3:
            random_score += 1

        # Too many short "words" (< 2 chars)
        words = text.split()
        short_words = sum(1 for w in words if len(w) < 2)
        if words and short_words > len(words) * 0.5:
            random_score += 1

        # Too many random character combinations
        gibberish_patterns = re.findall(r'[a-z]{1}\d+[a-z]{1,3}\d+', text, re.I)
        if len(gibberish_patterns) > 3:
            random_score += 1

        # If multiple random indicators, likely garbled
        return random_score < 2

    def filter_quality_endpoints(self,
                                endpoints: List[Dict],
                                verbose: bool = False) -> List[Dict]:
        """Filter endpoints by quality, removing broken ones.

        Args:
            endpoints: List of endpoint dicts
            verbose: Print quality test results

        Returns:
            Filtered list of quality endpoints
        """
        quality_endpoints = []

        for i, endpoint in enumerate(endpoints, 1):
            if verbose:
                print(f"Testing endpoint {i}/{len(endpoints)}: {endpoint['url']}")

            score, results = self.check_endpoint_quality(endpoint)

            endpoint['quality_score'] = score
            endpoint['quality_test_results'] = results
            endpoint['quality_tested_at'] = time.time()

            if score >= self.min_quality_score:
                quality_endpoints.append(endpoint)
                if verbose:
                    print(f"  ✅ PASS (score: {score:.2f})")
            else:
                if verbose:
                    print(f"  ❌ FAIL (score: {score:.2f}) - {results.get('error', 'quality too low')}")

        return quality_endpoints


def validate_endpoint_quality(endpoint: Dict,
                              timeout: float = 10.0,
                              min_score: float = 0.5) -> Tuple[bool, float, Dict]:
    """Validate a single endpoint's quality.

    Args:
        endpoint: Endpoint dict to test
        timeout: Request timeout
        min_score: Minimum quality score to pass

    Returns:
        Tuple of (passed, score, test_results)
    """
    checker = EndpointQualityChecker(timeout=timeout, min_quality_score=min_score)
    score, results = checker.check_endpoint_quality(endpoint)

    passed = score >= min_score
    return passed, score, results


def filter_quality_endpoints(endpoints: List[Dict],
                            min_score: float = 0.5,
                            timeout: float = 10.0,
                            verbose: bool = False) -> List[Dict]:
    """Filter list of endpoints, keeping only quality ones.

    Args:
        endpoints: List of endpoints to filter
        min_score: Minimum quality score (0.0-1.0)
        timeout: Test timeout per endpoint
        verbose: Print test results

    Returns:
        Filtered list of quality endpoints
    """
    checker = EndpointQualityChecker(timeout=timeout, min_quality_score=min_score)
    return checker.filter_quality_endpoints(endpoints, verbose=verbose)
