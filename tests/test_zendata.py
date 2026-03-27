"""
Tests for Zendata DLP evaluation (integrations/zendata_integration.py).

Covers: skip when unconfigured, mock fallback on connection error,
PII detection (SSN, email), risk level calculation, trust boundary,
and classification labels.
"""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from ai_telemetry.client import TelemetryEvent
from ai_telemetry.integrations.zendata_integration import _mock_evaluate, evaluate_event


def _event(**kwargs):
    return TelemetryEvent(event_type="llm_call", **kwargs)


class TestEvaluateEvent:
    def test_skips_silently_when_engine_url_not_set(self, monkeypatch):
        monkeypatch.delenv("ZENDATA_ENGINE_URL", raising=False)
        event = _event()
        evaluate_event(event)
        assert event.pii_detected is None  # untouched

    def test_uses_mock_evaluate_when_engine_url_is_mock(self, monkeypatch):
        monkeypatch.setenv("ZENDATA_ENGINE_URL", "mock")
        event = _event(
            input_messages=[{"role": "user", "content": "hello"}],
        )
        evaluate_event(event)
        # _mock_evaluate must have run and set these fields
        assert event.pii_detected is not None
        assert event.risk_signal is not None

    def test_falls_back_to_mock_on_connection_error(self, monkeypatch):
        monkeypatch.setenv("ZENDATA_ENGINE_URL", "http://does-not-exist.test")
        monkeypatch.setenv("ZENDATA_API_KEY", "key")
        event = _event(input_messages=[{"role": "user", "content": "hi"}])
        evaluate_event(event)  # must not raise
        assert event.risk_signal is not None

    def test_enriches_event_from_http_response(self, monkeypatch):
        import json
        monkeypatch.setenv("ZENDATA_ENGINE_URL", "http://zendata.test/eval")
        monkeypatch.setenv("ZENDATA_API_KEY", "key")

        mock_resp_data = json.dumps({
            "pii_detected": True,
            "pii_types": "SSN",
            "trust_boundary_crossed": False,
            "risk_signal": "HIGH",
            "zendata_composite_risk": "65",
            "zendata_classification": "Confidential",
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_resp_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            event = _event(
                input_messages=[{"role": "user", "content": "test"}],
            )
            evaluate_event(event)

        assert event.pii_detected is True
        assert event.pii_types == "SSN"
        assert event.risk_signal == "HIGH"
        assert event.zendata_classification == "Confidential"


class TestMockEvaluate:
    def test_no_pii_in_clean_content(self):
        event = _event(
            input_messages=[{"role": "user", "content": "what is the weather today?"}],
        )
        _mock_evaluate(event)
        assert event.pii_detected is False
        assert event.pii_types == ""

    def test_detects_ssn_pii(self):
        event = _event(
            input_messages=[{"role": "user", "content": "my ssn is 123-45-6789"}],
        )
        _mock_evaluate(event)
        assert event.pii_detected is True
        assert "SSN" in event.pii_types

    def test_detects_social_security_pii(self):
        event = _event(
            input_messages=[{"role": "user", "content": "social security number"}],
        )
        _mock_evaluate(event)
        assert event.pii_detected is True
        assert "SSN" in event.pii_types

    def test_detects_email_pii(self):
        event = _event(
            input_messages=[{"role": "user", "content": "contact admin@example.com"}],
        )
        _mock_evaluate(event)
        assert event.pii_detected is True
        assert "Email" in event.pii_types

    def test_detects_both_pii_types(self):
        event = _event(
            input_messages=[{
                "role": "user",
                "content": "ssn: 123-45-6789, email: foo@bar.com",
            }],
        )
        _mock_evaluate(event)
        assert "SSN" in event.pii_types
        assert "Email" in event.pii_types

    def test_low_risk_no_pii_no_boundary(self):
        event = _event(sailpoint_risk="10")
        _mock_evaluate(event)
        assert event.risk_signal == "LOW"
        assert event.zendata_classification == "Internal"
        assert int(event.zendata_composite_risk) == 10

    def test_high_risk_with_pii(self):
        # base=10, +30 for PII => 40 < 50 → still LOW
        # base=30 + PII=30 => 60 → HIGH
        event = _event(
            sailpoint_risk="30",
            input_messages=[{"role": "user", "content": "my email is x@y.com"}],
        )
        _mock_evaluate(event)
        assert event.risk_signal == "HIGH"
        assert event.zendata_classification == "Confidential"

    def test_critical_risk_with_pii_and_trust_boundary(self):
        # base=20 + PII=30 + boundary=40 => 90 → CRITICAL
        event = _event(
            sailpoint_risk="20",
            input_messages=[{"role": "user", "content": "ssn: 111-22-3333 external"}],
        )
        _mock_evaluate(event)
        assert event.risk_signal == "CRITICAL"
        assert event.zendata_classification == "Highly Confidential"
        assert int(event.zendata_composite_risk) <= 100

    def test_trust_boundary_crossed_on_external_content(self):
        event = _event(
            input_messages=[{"role": "user", "content": "send to external partner"}],
        )
        _mock_evaluate(event)
        assert event.trust_boundary_crossed is True

    def test_trust_boundary_crossed_on_send_email_tool(self):
        event = _event(tools_called="send_email|search")
        _mock_evaluate(event)
        assert event.trust_boundary_crossed is True

    def test_no_trust_boundary_for_internal_content(self):
        event = _event(
            input_messages=[{"role": "user", "content": "internal report summary"}],
        )
        _mock_evaluate(event)
        assert event.trust_boundary_crossed is False

    def test_composite_risk_capped_at_100(self):
        # base=90 + PII=30 + boundary=40 would be 160 → capped at 100
        event = _event(
            sailpoint_risk="90",
            input_messages=[{"role": "user", "content": "ssn: 111-22-3333 external"}],
        )
        _mock_evaluate(event)
        assert int(event.zendata_composite_risk) <= 100

    def test_invalid_sailpoint_risk_defaults_to_10(self):
        event = _event(sailpoint_risk="not-a-number")
        _mock_evaluate(event)
        # base=0 (ValueError caught) → composite=0 → LOW
        assert event.risk_signal == "LOW"
