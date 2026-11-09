import json

import pytest

from src import config, llm
from src.agent import Agent, _total_chars


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, call_id, name, arguments="{}"):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        data = {"role": "assistant"}
        if self.content is not None or not exclude_none:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in self.tool_calls
            ]
        return data


@pytest.fixture
def fake_llm(monkeypatch):
    state = {"replies": []}

    def fake_chat(messages, tools=None):
        if not state["replies"]:
            raise AssertionError("fake LLM ran out of replies")
        reply = state["replies"].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm, "chat", fake_chat)
    return state


def make_agent(**kw):
    return Agent(system_prompt=kw.pop("system_prompt", "test system"))


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_plain_reply(fake_llm):
    fake_llm["replies"] = [FakeMessage("Hello there.")]
    agent = make_agent()
    assert agent.run_turn("hi") == "Hello there."
    assert agent.history[-1] == {"role": "assistant", "content": "Hello there."}
    assert agent.last_tools == []
    json.dumps(agent.history)


def test_tool_round_trip(fake_llm):
    fake_llm["replies"] = [
        FakeMessage(None, [FakeCall("call_1", "get_time", "{}")]),
        FakeMessage("It is noon."),
    ]
    agent = make_agent()
    assert agent.run_turn("what time is it?") == "It is noon."
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert not tool_msgs[0]["content"].startswith("ERROR")
    assert agent.last_tools[0]["name"] == "get_time"
    assert agent.last_tools[0]["ms"] >= 0


def test_parallel_tool_calls_each_get_a_result(fake_llm):
    fake_llm["replies"] = [
        FakeMessage(
            None,
            [
                FakeCall("a", "get_time", "{}"),
                FakeCall("b", "list_files", '{"path": "."}'),
            ],
        ),
        FakeMessage("done"),
    ]
    agent = make_agent()
    agent.run_turn("do two things")
    ids = [m["tool_call_id"] for m in agent.history if m["role"] == "tool"]
    assert ids == ["a", "b"]
    assert len(agent.last_tools) == 2


def test_step_limit_appends_assistant_message(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_ROUNDS", 3)
    fake_llm["replies"] = [FakeMessage(None, [FakeCall("x", "get_time", "{}")]) for _ in range(3)]
    agent = make_agent()
    reply = agent.run_turn("loop forever")
    assert "step limit" in reply
    assert agent.history[-1] == {"role": "assistant", "content": reply}
    assert len([m for m in agent.history if m["role"] == "tool"]) == 3
    json.dumps(agent.history)


def test_unknown_tool_returns_error_and_turn_survives(fake_llm):
    fake_llm["replies"] = [
        FakeMessage(None, [FakeCall("x", "invented_tool", "{}")]),
        FakeMessage("Recovered."),
    ]
    agent = make_agent()
    assert agent.run_turn("use a bad tool") == "Recovered."
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    assert tool_msg["content"].startswith("ERROR: unknown tool")


@pytest.mark.parametrize(
    "args",
    [
        {"fliepath": "x.gd"},
        {"path": "x.gd", "extra": 1},
    ],
)
def test_bad_argument_names_return_error(fake_llm, args):
    fake_llm["replies"] = [
        FakeMessage(None, [FakeCall("x", "read_file", json.dumps(args))]),
        FakeMessage("Recovered."),
    ]
    agent = make_agent()
    assert agent.run_turn("bad args") == "Recovered."
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    assert tool_msg["content"].startswith("ERROR: bad arguments")


def test_missing_required_argument(fake_llm):
    fake_llm["replies"] = [
        FakeMessage(None, [FakeCall("x", "read_file", "{}")]),
        FakeMessage("Recovered."),
    ]
    agent = make_agent()
    assert agent.run_turn("no args") == "Recovered."
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    assert tool_msg["content"].startswith("ERROR: bad arguments")


def test_broken_json_arguments(fake_llm):
    fake_llm["replies"] = [
        FakeMessage(None, [FakeCall("x", "read_file", "{not json")]),
        FakeMessage("Recovered."),
    ]
    agent = make_agent()
    assert agent.run_turn("broken json") == "Recovered."
    tool_msg = next(m for m in agent.history if m["role"] == "tool")
    assert tool_msg["content"] == "ERROR: could not parse tool arguments"
    assert agent.last_tools[0]["args"] == "{not json"


def test_tool_output_budget_is_enforced(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CHARS_PER_TURN", 10)
    fake_llm["replies"] = [
        FakeMessage(
            None,
            [
                FakeCall("a", "get_time", "{}"),
                FakeCall("b", "get_time", "{}"),
            ],
        ),
        FakeMessage("ok"),
    ]
    agent = make_agent()
    agent.run_turn("time twice")
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    assert len(tool_msgs[0]["content"]) <= 10
    assert tool_msgs[1]["content"] == "ERROR: tool output budget used up"


def test_llm_exception_propagates(fake_llm):
    fake_llm["replies"] = [RuntimeError("network down")]
    agent = make_agent()
    with pytest.raises(RuntimeError):
        agent.run_turn("hello")


def test_abort_closes_the_turn(fake_llm):
    agent = make_agent()
    agent.history.append({"role": "user", "content": "hi"})
    agent.abort("Sorry.")
    assert agent.history[-1] == {"role": "assistant", "content": "Sorry."}


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def add_tool_turn(agent, tools_count=2, size=0):
    agent.history.append({"role": "user", "content": "user turn"})
    calls = [
        {
            "id": f"c{len(agent.history)}_{i}",
            "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        }
        for i in range(tools_count)
    ]
    agent.history.append({"role": "assistant", "content": None, "tool_calls": calls})
    for call in calls:
        agent.history.append(
            {"role": "tool", "tool_call_id": call["id"], "content": "x" * size}
        )


def add_plain_turn(agent):
    agent.history.append({"role": "user", "content": "u"})
    agent.history.append({"role": "assistant", "content": "a"})


def assert_trimmed_invariants(agent):
    assert agent.history[0]["role"] == "system"
    assert agent.history[1]["role"] == "user"
    assert len([m for m in agent.history if m["role"] == "user"]) <= config.MAX_HISTORY_TURNS
    # Every tool message is preceded by the assistant message that requested it.
    pending = set()
    for msg in agent.history[1:]:
        if msg["role"] == "assistant":
            pending = {c["id"] for c in msg.get("tool_calls") or []}
        elif msg["role"] == "tool":
            assert msg["tool_call_id"] in pending
            pending.discard(msg["tool_call_id"])


def test_trim_keeps_last_12_turns():
    agent = make_agent()
    for _ in range(20):
        add_plain_turn(agent)
    agent._trim()
    assert_trimmed_invariants(agent)
    assert len([m for m in agent.history if m["role"] == "user"]) == 12


def test_trim_never_splits_a_tool_group(monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 1)
    agent = make_agent()
    for _ in range(3):
        add_tool_turn(agent)
    agent._trim()
    assert_trimmed_invariants(agent)
    assert len([m for m in agent.history if m["role"] == "user"]) == 1


def test_trim_character_cap_drops_whole_turns():
    agent = make_agent()
    for _ in range(12):
        add_tool_turn(agent, tools_count=1, size=16000)
    assert _total_chars(agent.history[1:]) > config.MAX_HISTORY_CHARS
    agent._trim()
    assert_trimmed_invariants(agent)
    assert _total_chars(agent.history[1:]) <= config.MAX_HISTORY_CHARS


def test_trim_keeps_one_oversized_turn_whole():
    agent = make_agent()
    agent.history.append({"role": "user", "content": "many calls"})
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        }
        for i in range(45)
    ]
    agent.history.append({"role": "assistant", "content": None, "tool_calls": calls})
    for call in calls:
        agent.history.append({"role": "tool", "tool_call_id": call["id"], "content": "x"})
    assert len(agent.history[1:]) > config.MAX_HISTORY_MESSAGES
    agent._trim()  # must terminate and keep the turn
    assert len(agent.history[1:]) > config.MAX_HISTORY_MESSAGES
    assert agent.history[1]["role"] == "user"


def test_trim_after_a_full_turn(fake_llm, monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 1)
    agent = make_agent()
    add_tool_turn(agent)
    fake_llm["replies"] = [FakeMessage("ok")]
    agent.run_turn("new question")
    assert_trimmed_invariants(agent)
