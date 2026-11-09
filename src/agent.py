"""The tool-calling loop and conversation history."""

import json
import time

from . import config, llm, memory, tools


def _total_chars(msgs) -> int:
    total = 0
    for m in msgs:
        content = m.get("content")
        if content is None:
            total += 0
        elif isinstance(content, str):
            total += len(content)
        else:
            total += len(json.dumps(content))
        calls = m.get("tool_calls")
        if calls:
            total += len(json.dumps(calls))
    return total


def _summarize_args(args):
    """Keep turns.jsonl small when the model writes a large file."""
    if not isinstance(args, dict):
        text = str(args)
        return text[:500] + ("... [truncated]" if len(text) > 500 else "")
    out = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 500:
            out[key] = value[:500] + f"... [{len(value)} chars]"
        else:
            out[key] = value
    return out


class Agent:
    def __init__(self, system_prompt: str | None = None):
        if system_prompt is None:
            system_prompt = config.build_system_prompt(memory.load_notes())
        self.history = [{"role": "system", "content": system_prompt}]
        self.last_tools: list[dict] = []

    def run_turn(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        self.last_tools = []
        tool_chars = 0

        for _ in range(config.MAX_TOOL_ROUNDS):
            if config.DEBUG:
                json.dumps(self.history)  # history must stay JSON-serializable
            msg = llm.chat(self.history, tools.SCHEMAS)
            self.history.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                self._trim()
                return msg.content or ""

            for call in msg.tool_calls:
                started = time.perf_counter()
                name = call.function.name
                raw_args = call.function.arguments or "{}"
                parsed = None
                try:
                    args = json.loads(raw_args)
                    parsed = args
                    result = tools.run(name, args)
                except json.JSONDecodeError:
                    result = "ERROR: could not parse tool arguments"

                left = config.MAX_TOOL_CHARS_PER_TURN - tool_chars
                if left <= 0:
                    result = "ERROR: tool output budget used up"
                else:
                    result = result[: min(config.MAX_TOOL_RESULT_CHARS, left)]
                    tool_chars += len(result)

                self.last_tools.append(
                    {
                        "name": name,
                        "args": _summarize_args(parsed if parsed is not None else raw_args),
                        "ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self.history.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )

        # Keep history ending in an assistant message before the next user turn.
        reply = "I hit my step limit. Try rephrasing."
        self.history.append({"role": "assistant", "content": reply})
        self._trim()
        return reply

    def abort(self, reply: str):
        """Close a failed turn so the next user message does not follow a tool reply."""
        self.history.append({"role": "assistant", "content": reply})
        self._trim()

    def _trim(self):
        sys_msg, rest = self.history[0], self.history[1:]

        # Every user message starts a turn; cut only at those boundaries.
        starts = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        if len(starts) > config.MAX_HISTORY_TURNS:
            rest = rest[starts[-config.MAX_HISTORY_TURNS] :]

        # Hard caps by message count and by size, again at user boundaries.
        while (
            len(rest) > config.MAX_HISTORY_MESSAGES
            or _total_chars(rest) > config.MAX_HISTORY_CHARS
        ):
            nxt = next(
                (i for i, m in enumerate(rest[1:], 1) if m.get("role") == "user"),
                None,
            )
            if nxt is None:
                # One turn alone can exceed a cap (many parallel tool calls).
                # Without this break the loop would never end.
                break
            rest = rest[nxt:]

        # Guard against future refactors that might cut mid-turn.
        assert not rest or rest[0].get("role") == "user", "trim left history starting mid-turn"
        self.history = [sys_msg] + rest

    def reset(self):
        self.history = [{"role": "system", "content": config.build_system_prompt(memory.load_notes())}]
        self.last_tools = []
