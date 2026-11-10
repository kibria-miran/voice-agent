import os
import sys
import time
from pathlib import Path

import pytest

from src import config, memory, tools


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(config, "WORKSPACE", ws)
    return ws


# ---------------------------------------------------------------------------
# safe_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["../x", "/etc/passwd", "a/../../outside", "../../etc"],
)
def test_safe_path_rejects_escapes(workspace, bad):
    with pytest.raises(ValueError):
        tools.safe_path(bad)


def test_safe_path_rejects_symlink_pointing_out(workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = workspace / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        tools.safe_path("link.txt")


def test_safe_path_accepts_absolute_path_inside(workspace):
    target = workspace / "sub" / "a.txt"
    target.parent.mkdir()
    target.write_text("ok")
    assert tools.safe_path(str(target)) == target.resolve()


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------


def test_write_read_roundtrip(workspace):
    result = tools.write_file("sub/player.gd", "extends CharacterBody2D\n")
    assert isinstance(result, str) and "Wrote" in result
    assert (workspace / "sub" / "player.gd").read_text() == "extends CharacterBody2D\n"
    assert "extends CharacterBody2D" in tools.read_file("sub/player.gd")


def test_write_refuses_symlink_target(workspace):
    real = workspace / "real.txt"
    real.write_text("original")
    link = workspace / "link.txt"
    link.symlink_to(real)
    result = tools.write_file("link.txt", "overwritten")
    assert result.startswith("ERROR:")
    assert real.read_text() == "original"


def test_read_binary_is_an_error(workspace):
    (workspace / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert tools.read_file("blob.bin").startswith("ERROR: binary")


def test_read_missing_file_is_an_error(workspace):
    assert tools.read_file("nope.txt").startswith("ERROR:")


def test_write_creates_parent_dirs(workspace):
    tools.write_file("a/b/c.txt", "x")
    assert (workspace / "a" / "b" / "c.txt").is_file()


def test_list_files_truncates_at_200(workspace):
    for i in range(205):
        (workspace / f"f{i:03d}.txt").write_text("x")
    out = tools.list_files(".")
    lines = out.splitlines()
    assert lines[-1] == "... and 5 more"
    assert len(lines) == 201


def test_list_files_empty(workspace):
    assert tools.list_files(".") == "(empty directory)"


def test_list_files_rejects_file(workspace):
    (workspace / "a.txt").write_text("x")
    assert tools.list_files("a.txt").startswith("ERROR: not a directory")


def test_cat_large_file_tells_model_to_use_read_file(workspace):
    (workspace / "big.txt").write_text("x" * (config.READ_FILE_BYTES + 10))
    out = tools.run_command("cat big.txt")
    assert out.startswith("ERROR:") and "read_file" in out


# ---------------------------------------------------------------------------
# run_command policy
# ---------------------------------------------------------------------------


def test_read_only_runs_without_prompt(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    out = tools.run_command("pwd")
    assert "ERROR" not in out
    assert str(workspace.resolve()) in out


def test_shell_features_rejected(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    assert tools.run_command("ls | wc").startswith("ERROR: shell features")
    assert tools.run_command("ls > out.txt").startswith("ERROR: shell features")
    assert tools.run_command("ls && pwd").startswith("ERROR: shell features")


def test_quoted_semicolons_are_allowed_in_exec_commands(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: True)
    out = tools.run_command(f'{sys.executable} -c "import sys; print(42)"')
    assert out.strip() == "42"


def test_sudo_refused_without_prompt(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    assert tools.run_command("sudo ls").startswith("ERROR: sudo")
    assert tools.run_command("/usr/bin/sudo ls").startswith("ERROR: sudo")


def test_safe_git_form_does_not_prompt(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    out = tools.run_command("git --no-pager status")
    assert not out.startswith("ERROR: command refused")


def test_other_git_commands_always_ask(workspace, monkeypatch):
    called = []
    monkeypatch.setattr(tools, "_confirm", lambda argv: called.append(argv) or False)
    out = tools.run_command("git -c core.pager=cat status")
    assert called and out.startswith("ERROR: command refused")


def test_outside_path_argument_asks(workspace, monkeypatch):
    called = []
    monkeypatch.setattr(tools, "_confirm", lambda argv: called.append(argv) or False)
    out = tools.run_command("cat /etc/hostname")
    assert called and out.startswith("ERROR: command refused")


def test_exec_class_command_asks_and_runs(workspace, monkeypatch):
    called = []
    monkeypatch.setattr(tools, "_confirm", lambda argv: called.append(argv) or True)
    out = tools.run_command(f'{sys.executable} -c "print(\'hi\')"')
    assert called and out.strip() == "hi"


def test_exec_class_denied(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: False)
    out = tools.run_command(f'{sys.executable} -c "print(1)"')
    assert out.startswith("ERROR: command refused")


def test_stderr_is_labeled(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: True)
    out = tools.run_command(f'{sys.executable} -c "import sys; sys.stderr.write(\'boom\')"')
    assert "[stderr]" in out and "boom" in out


def test_timeout_kills_the_process_group(workspace, monkeypatch):
    monkeypatch.setattr(config, "COMMAND_TIMEOUT_S", 0.5)
    monkeypatch.setattr(tools, "_confirm", lambda argv: True)
    code = (
        "import subprocess,time; "
        "p=subprocess.Popen(['sleep','30']); "
        "open('child.pid','w').write(str(p.pid)); "
        "time.sleep(30)"
    )
    started = time.time()
    out = tools.run_command(f'{sys.executable} -c "{code}"')
    assert "timed out" in out
    assert time.time() - started < 5

    pid_file = workspace / "child.pid"
    assert pid_file.is_file()
    pid = int(pid_file.read_text())
    deadline = time.time() + 3
    alive = True
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.05)
    if alive:  # do not leak a process if the assertion is about to fail
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    assert not alive, "child process survived the group kill"


def test_find_rejects_exec(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    assert "not allowed" in tools.run_command("find . -exec rm {} +")


def test_find_rejects_file_writing_actions(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    assert "not allowed" in tools.run_command("find . -fprintf out.txt %p")
    assert "not allowed" in tools.run_command("find . -fprint out.txt")


def test_option_value_outside_workspace_asks(workspace, monkeypatch):
    called = []
    monkeypatch.setattr(tools, "_confirm", lambda argv: called.append(argv) or False)
    out = tools.run_command("grep --file=/etc/passwd .")
    assert called and out.startswith("ERROR: command refused")


def test_option_value_inside_workspace_does_not_ask(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_confirm", lambda argv: (_ for _ in ()).throw(AssertionError))
    out = tools.run_command("grep --exclude=*.py -r x .")
    assert not out.startswith("ERROR: command refused")


# ---------------------------------------------------------------------------
# Dispatcher and misc tools
# ---------------------------------------------------------------------------


def test_dispatcher_unknown_tool(workspace):
    assert tools.run("no_such_tool", {}).startswith("ERROR: unknown tool")


def test_dispatcher_bad_argument_names(workspace):
    out = tools.run("read_file", {"fliepath": "x"})
    assert out.startswith("ERROR: bad arguments")


def test_dispatcher_bad_arg_types(workspace):
    assert tools.run("read_file", "not-a-dict").startswith("ERROR: bad arguments")


def test_every_tool_returns_a_string(workspace):
    assert isinstance(tools.read_file("missing"), str)
    assert isinstance(tools.list_files("missing"), str)
    assert isinstance(tools.write_file("x.txt", "x"), str)
    assert isinstance(tools.godot_check("missing.gd"), str)
    assert isinstance(tools.get_time(), str)
    assert isinstance(tools.remember("a fact"), str)


def test_get_time(workspace):
    out = tools.get_time()
    assert ":" in out and out.count(",") >= 1


def test_remember_writes_notes(workspace):
    out = tools.remember("uses Godot 4.3")
    assert "Remembered" in out
    assert "uses Godot 4.3" in (workspace / "notes.md").read_text()
    assert "uses Godot 4.3" in memory.load_notes()


def test_godot_check_without_godot(workspace, monkeypatch):
    (workspace / "player.gd").write_text("extends Node\n")
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    assert tools.godot_check("player.gd").startswith("ERROR: godot not found")


def test_godot_check_rejects_non_gd(workspace):
    (workspace / "a.txt").write_text("x")
    assert tools.godot_check("a.txt").startswith("ERROR: not a .gd")


def test_read_file_truncates(workspace):
    (workspace / "long.txt").write_text("y" * (config.READ_FILE_BYTES + 100))
    out = tools.read_file("long.txt")
    assert "truncated" in out
    assert len(out) < config.READ_FILE_BYTES + 200


def test_write_too_large(workspace, monkeypatch):
    monkeypatch.setattr(config, "MAX_WRITE_CHARS", 10)
    assert tools.write_file("x.txt", "12345678901").startswith("ERROR: content too large")
