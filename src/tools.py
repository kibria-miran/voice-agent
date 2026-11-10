"""Tool functions, JSON schemas, and the guarded dispatcher.

Every tool returns a string: either human-readable output or ``ERROR: ...``.
Tools never raise to the agent loop.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from . import config, memory

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def safe_path(p: str) -> Path:
    """Resolve ``p`` under WORKSPACE and refuse anything that escapes it."""
    if not isinstance(p, str):
        raise ValueError("path must be a string")
    p = p.strip()
    if not p:
        raise ValueError("path is empty")
    if "\x00" in p:
        raise ValueError("path contains a NUL byte")
    workspace = Path(config.WORKSPACE).resolve()
    full = (workspace / p).resolve()  # resolve follows symlinks, catching escapes
    if full != workspace and workspace not in full.parents:
        raise ValueError(f"path outside workspace: {p}")
    return full


def _display(path: Path) -> str:
    workspace = Path(config.WORKSPACE).resolve()
    try:
        rel = path.resolve().relative_to(workspace)
    except ValueError:
        return str(path)
    return "." if str(rel) == "." else str(rel)


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


def list_files(path: str = ".") -> str:
    full = safe_path(path)
    if not full.exists():
        return f"ERROR: no such directory: {path}"
    if not full.is_dir():
        return f"ERROR: not a directory: {path}"
    try:
        entries = sorted(full.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    total = len(entries)
    lines = []
    for entry in entries[: config.LIST_MAX_ENTRIES]:
        suffix = "/" if entry.is_dir() else ""
        try:
            size = "" if entry.is_dir() else f"  ({entry.stat().st_size} B)"
        except OSError:
            size = ""
        lines.append(f"{entry.name}{suffix}{size}")
    if total > config.LIST_MAX_ENTRIES:
        lines.append(f"... and {total - config.LIST_MAX_ENTRIES} more")
    return "\n".join(lines) if lines else "(empty directory)"


def read_file(path: str) -> str:
    full = safe_path(path)
    if not full.exists():
        return f"ERROR: no such file: {path}"
    if not full.is_file():
        return f"ERROR: not a regular file: {path}"
    try:
        with full.open("rb") as fh:
            raw = fh.read(config.READ_FILE_BYTES + 1)
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    if b"\x00" in raw:
        return f"ERROR: binary file, not text: {path}"
    text = raw.decode("utf-8", errors="replace")
    truncated = len(raw) > config.READ_FILE_BYTES
    if truncated:
        text = text[: config.READ_FILE_BYTES]
        text += f"\n... [truncated at {config.READ_FILE_BYTES} bytes]"
    return text


def write_file(path: str, content: str) -> str:
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if len(content) > config.MAX_WRITE_CHARS:
        return (
            f"ERROR: content too large ({len(content)} chars, "
            f"max {config.MAX_WRITE_CHARS})"
        )
    workspace = Path(config.WORKSPACE)
    raw = workspace / path if not Path(path).is_absolute() else Path(path)
    # safe_path resolves the symlink, so check the original target first.
    if raw.is_symlink():
        return f"ERROR: refusing to write through symlink: {path}"
    full = safe_path(path)
    if full.exists() and full.is_dir():
        return f"ERROR: path is a directory: {path}"
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return f"Wrote {len(content)} chars to {_display(full)}"


# ---------------------------------------------------------------------------
# run_command policy
# ---------------------------------------------------------------------------

READ_ONLY_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "grep",
    "wc",
    "pwd",
    "tree",
    "echo",
    "find",
}
FIND_FORBIDDEN = {
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-delete",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-fls",
}
SAFE_GIT_FORMS = (
    ("git", "--no-pager", "status"),
    ("git", "--no-pager", "diff", "--stat", "--no-ext-diff"),
    ("git", "--no-pager", "log", "--oneline", "-20"),
)
PRIVILEGE_COMMANDS = {"sudo", "sudoedit", "su", "doas", "pkexec", "run0"}


def _has_unquoted_meta(cmd: str) -> bool:
    """True when a shell operator appears outside quotes.

    Quoted semicolons (e.g. inside a Python one-liner) are harmless because
    we never use a shell; unquoted pipes/redirects signal model intent to use
    shell features, which we reject.
    """
    quote = None
    i = 0
    while i < len(cmd):
        ch = cmd[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 2
                continue
            if ch == quote:
                quote = None
        else:
            if ch in ("'", '"'):
                quote = ch
            elif ch in "|&;<>`\n\r":
                return True
        i += 1
    return False


def _outside_arg(argv) -> str | None:
    """Return the first argument that resolves outside WORKSPACE.

    Both bare arguments and the value half of ``--option=value`` are checked,
    so ``grep --file=/etc/passwd ...`` cannot read outside without a prompt.
    """
    workspace = Path(config.WORKSPACE).resolve()
    for token in argv[1:]:
        if not token:
            continue
        candidates = [token]
        if "=" in token:
            value = token.split("=", 1)[1]
            if value:
                candidates.append(value)
        for candidate in candidates:
            try:
                path = Path(candidate)
                full = (
                    path.resolve()
                    if path.is_absolute()
                    else (workspace / path).resolve()
                )
            except (OSError, ValueError):
                continue
            if full != workspace and workspace not in full.parents:
                return candidate
    return None


def _confirm(argv) -> bool:
    """Typed terminal confirmation.  Tests monkeypatch this function."""
    print("\n[voice-agent] About to run:", " ".join(shlex.quote(a) for a in argv))
    print(f"[voice-agent] Working directory: {config.WORKSPACE}")
    try:
        answer = input("Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _run_argv(argv, timeout: float | None = None) -> str:
    timeout = config.COMMAND_TIMEOUT_S if timeout is None else timeout
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    proc = subprocess.Popen(
        argv,
        cwd=config.WORKSPACE,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.communicate(timeout=2)  # drain pipes, but never hang
        except subprocess.TimeoutExpired:
            proc.kill()  # something escaped the group and holds the pipes
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
        return f"ERROR: command timed out after {timeout}s"
    combined = out + ("\n[stderr]\n" + err if err.strip() else "")
    return combined[: config.MAX_TOOL_RESULT_CHARS] or "(no output)"


def run_command(cmd: str) -> str:
    if not isinstance(cmd, str):
        raise ValueError("cmd must be a string")
    if not cmd.strip():
        return "ERROR: empty command"
    if _has_unquoted_meta(cmd):
        return "ERROR: shell features are not allowed (no pipes, redirects, &&, ;, etc.)"
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return f"ERROR: could not parse command: {exc}"
    if not argv:
        return "ERROR: empty command"
    if Path(argv[0]).name in PRIVILEGE_COMMANDS:
        return "ERROR: sudo/su/doas/pkexec are refused"

    if argv[0] == "find":
        bad = next((a for a in argv[1:] if a in FIND_FORBIDDEN), None)
        if bad:
            return f"ERROR: find option {bad} is not allowed"
        outside = _outside_arg(argv)
        if outside and not _confirm(argv):
            return "ERROR: command refused by user"
        return _run_argv(argv)

    if tuple(argv) in SAFE_GIT_FORMS:
        return _run_argv(argv)

    if argv[0] in READ_ONLY_COMMANDS:
        outside = _outside_arg(argv)
        if outside:
            if not _confirm(argv):
                return "ERROR: command refused by user"
        if argv[0] == "cat":
            for token in argv[1:]:
                if token.startswith("-"):
                    continue
                candidate = Path(token)
                if not candidate.is_absolute():
                    candidate = Path(config.WORKSPACE) / candidate
                try:
                    if candidate.is_file() and candidate.stat().st_size > config.READ_FILE_BYTES:
                        return (
                            f"ERROR: {token} is larger than {config.READ_FILE_BYTES} bytes; "
                            "use read_file instead"
                        )
                except OSError:
                    pass
        return _run_argv(argv)

    # Anything that executes code always asks, in the terminal, every time.
    if not _confirm(argv):
        return "ERROR: command refused by user"
    return _run_argv(argv)


# ---------------------------------------------------------------------------
# Godot + misc tools
# ---------------------------------------------------------------------------


def godot_check(script_path: str) -> str:
    full = safe_path(script_path)
    if full.suffix != ".gd":
        return f"ERROR: not a .gd script: {script_path}"
    if not full.is_file():
        return f"ERROR: no such script: {script_path}"
    godot = shutil.which("godot")
    if not godot:
        return "ERROR: godot not found in PATH"
    return _run_argv(
        [godot, "--headless", "--check-only", "--script", str(full)],
        timeout=config.COMMAND_TIMEOUT_S,
    )


def get_time() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")


def remember(fact: str) -> str:
    if not isinstance(fact, str):
        raise ValueError("fact must be a string")
    try:
        note = memory.append_note(fact)
    except ValueError as exc:
        return f"ERROR: {exc}"
    except OSError as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    return f"Remembered: {note}"


# ---------------------------------------------------------------------------
# Registry and schemas
# ---------------------------------------------------------------------------

REGISTRY = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "godot_check": godot_check,
    "get_time": get_time,
    "remember": remember,
}

_PATH_PROP = {"type": "string", "description": "Path relative to the workspace"}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List a directory inside the workspace. Use '.' for the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {**_PATH_PROP, "default": "."}},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the workspace (truncated near 8 KB).",
            "parameters": {
                "type": "object",
                "properties": {"path": _PATH_PROP},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file inside the workspace. Use this for code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": _PATH_PROP,
                    "content": {"type": "string", "description": "Full file contents"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run ONE shell-free command (argv only, no pipes/redirects). "
                "Read-only commands run directly; anything that executes code "
                "requires the user to type y in the terminal. sudo is refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Single command, e.g. 'ls -la' or 'python script.py'",
                    }
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "godot_check",
            "description": "Validate one .gd script with godot --headless --check-only. No prompt.",
            "parameters": {
                "type": "object",
                "properties": {"script_path": _PATH_PROP},
                "required": ["script_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Append a short fact to the persistent notes file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "One short fact to remember"}
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
        },
    },
]


def run(name: str, args: dict) -> str:
    """Guarded dispatcher: unknown tools, bad args and bad JSON become ERROR strings."""
    fn = REGISTRY.get(name)
    if fn is None:
        return f"ERROR: unknown tool {name!r}"
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return f"ERROR: bad arguments for {name}: expected an object"
    try:
        result = fn(**args)
    except TypeError as exc:  # wrong argument names or arity
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - never raise into the agent loop
        return f"ERROR: {type(exc).__name__}: {exc}"
    if not isinstance(result, str):
        return json.dumps(result, ensure_ascii=False)
    return result
