"""Offline tests for safe, user-facing API error diagnostics."""
import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class FakeAPIError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class APIErrorTests(unittest.TestCase):
    def test_http_429_usage_limit_has_clear_quota_message(self):
        module = importlib.import_module('gift_cards.sentiment')
        self.assertTrue(hasattr(module, 'safe_api_error'), 'Safe API error mapper must exist')
        result = module.safe_api_error(FakeAPIError(429, 'The usage limit has been reached'))
        self.assertEqual(result, {
            'type': 'FakeAPIError',
            'http_status': 429,
            'code': 'API_USAGE_LIMIT_REACHED',
            'message': ('The request could not run because the API usage limit was reached. '
                        'Resolve the provider account quota, credits, or subscription usage cap before retrying.'),
        })

    def test_non_usage_429_is_described_as_rate_limit(self):
        module = importlib.import_module('gift_cards.sentiment')
        result = module.safe_api_error(FakeAPIError(429, 'Too many requests; try again later'))
        self.assertEqual(result['code'], 'API_RATE_LIMITED')
        self.assertIn('rate limit', result['message'].lower())

    def test_diagnostic_does_not_echo_exception_or_secret(self):
        module = importlib.import_module('gift_cards.sentiment')
        secret = 'sk-DO-NOT-PRINT'
        result = module.safe_api_error(FakeAPIError(401, f'Authorization Bearer {secret}'))
        self.assertNotIn(secret, str(result))


if __name__ == '__main__':
    unittest.main()
