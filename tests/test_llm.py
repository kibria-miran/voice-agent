"""LLM wrapper tests that never touch the network."""

import pytest

from src import config, llm


def test_self_test_tools_warns_instead_of_raising(monkeypatch, capsys):
    def boom(messages, schemas):
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(llm, "chat", boom)
    llm.self_test_tools([])  # must not raise
    assert "self-test failed" in capsys.readouterr().out


def test_self_test_tools_warns_when_model_does_not_call_a_tool(monkeypatch, capsys):
    class Msg:
        content = "It is noon."

    monkeypatch.setattr(llm, "chat", lambda messages, schemas: Msg())
    llm.self_test_tools([])
    assert "did not call a tool" in capsys.readouterr().out


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        llm._get_client()


def test_retryable_classification():
    assert llm._retryable(RuntimeError("nope")) is False

    status_exc = llm.APIStatusError.__new__(llm.APIStatusError)
    status_exc.status_code = 503
    assert llm._retryable(status_exc) is True
    status_exc.status_code = 400
    assert llm._retryable(status_exc) is False
