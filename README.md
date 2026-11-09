# Voice Agent

A push-to-talk voice helper for Godot 4 / GDScript work. Hold a key, speak,
release. The agent transcribes locally, sends the text to an NVIDIA NIM LLM
with tool calling, runs any tools it needs, and speaks a short answer back.

Plain Python, no agent framework.

```
[Mic] -> record (push-to-talk)
      -> STT: faster-whisper (tiny.en / base.en, int8, CPU)
      -> Agent loop: NVIDIA NIM (OpenAI-compatible) + tools
      -> TTS: Piper (ONNX, CPU)
      -> [Speaker]
```

## Hardware assumptions

- Dell Inspiron 3520, 4 GB RAM, no usable GPU.
- Arch Linux + XFCE on **X11**. `pynput` global keys do not work on Wayland.
- Target footprint: under ~800 MB RSS, under ~4 s from key release to first word
  (v1 non-streaming often lands around 3-6 s).

## Project layout

```
voice-agent/
├── .env                 # NVIDIA_API_KEY=... (never commit)
├── models/              # piper .onnx + .onnx.json
├── workspace/           # the only folder file tools may touch
├── logs/turns.jsonl     # one JSON line per turn
├── scripts/             # download_voice.py, check_env.py, chat_text.py
├── src/
│   ├── config.py        # settings, paths, limits
│   ├── audio_io.py      # persistent mic stream + chunked playback
│   ├── ptt.py           # PTTController + pynput wrapper
│   ├── stt.py           # faster-whisper wrapper
│   ├── tts.py           # piper wrapper + interruptible playback
│   ├── llm.py           # NIM client, retries, tool self-test
│   ├── tools.py         # tools, schemas, dispatcher, safety policy
│   ├── agent.py         # tool loop + history trimming
│   ├── memory.py        # notes.md helpers
│   └── main.py          # wiring
└── tests/
```

## Setup (Arch)

### 1. System packages

```bash
sudo pacman -S --needed python python-pip portaudio alsa-utils ffmpeg
```

### 2. Python environment

Python 3.11+ is what the code targets. If `pip install faster-whisper` or
`piper-tts` has no wheel for your Python version, use Python 3.12/3.13 in the
venv instead of the newest interpreter.

```bash
cd ~/voice-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# exact reproduction: pip install -r requirements.lock.txt
```

### 3. API key and model

```bash
cp .env.example .env
$EDITOR .env          # set NVIDIA_API_KEY
```

Get a key from <https://build.nvidia.com/>. Keep `NIM_MODEL` configurable:
tool-calling quality varies by model. The startup self-test sends a dummy tool
request and warns if the model answers in plain text instead of calling a tool.
If it warns, pick another tool-capable model from the catalog and update
`.env`.

Quick raw check:

```bash
curl https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"meta/llama-3.1-8b-instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
```

### 4. Piper voice

```bash
python scripts/download_voice.py --voice en_US-lessac --quality low
```

This writes `models/en_US-lessac-low.onnx` and its `.onnx.json` sidecar.
`config.PIPER_VOICE` points there by default.

Check the installed Piper CLI flags:

```bash
piper --help
```

`tts.speak` prefers the in-process `piper` Python API, which loads the voice
once at startup (a few seconds, ~150 MB RSS) and then synthesizes in roughly
1-3 s for a 2-3 s reply on this CPU. If that API is unavailable or fails, it
falls back to one CLI process per reply (`--output-raw` when available,
otherwise a temporary WAV). The CLI path reloads the voice every call and is
noticeably slower, so the in-process path is the default.

### 5. Phase 0 checks

```bash
python scripts/check_env.py
arecord -l
arecord -d 3 -f S16_LE -r 16000 test.wav && aplay test.wav
python -m sounddevice
```

Prefer an input device with a native 16000 Hz rate. If PortAudio has to
resample inside the callback thread, it costs CPU continuously and can drop
audio while Whisper runs.

On this laptop PortAudio's ALSA default output is HDMI, which may be silent
even though PipeWire is running. The app therefore prefers a `pipewire` or
`pulse` output device when one exists. To override it, list devices with
`python -m sounddevice` and set an index or name:

```bash
# in .env
AUDIO_OUTPUT_DEVICE=pipewire
# AUDIO_INPUT_DEVICE=pipewire
```

Enable zram (`zram-generator`) on a 4 GB machine.

Confirm your Godot version supports script checking:

```bash
godot --version
godot --help | grep -- --check-only
```

If `--check-only` is absent, remove or ignore the `godot_check` tool.

## Run

```bash
cd ~/voice-agent
source .venv/bin/activate
python -m src.main
```

Hold **F9**, speak, release. Press F9 during playback to stop the current
utterance. Ctrl+C exits cleanly.

The mic stream stays open for the whole session, so the hardware indicator
light may stay on. That is the tradeoff for avoiding per-press stream
open/close deadlocks.

### Debug without audio

```bash
python scripts/chat_text.py
```

This runs the same agent loop and tools from a text prompt, which is the
fastest way to check the API key, model tool-calling, and file tools before
touching microphone code.

### Measured on the target machine

The machine used for development was heavily loaded (2 cores, browsers open,
significant swap), so treat these as upper bounds:

| Stage | Observed |
| --- | --- |
| Whisper tiny.en load | ~6 s once |
| Whisper tiny.en, 2.4 s clip | ~3.5 s |
| Piper voice load | ~6-10 s once |
| Piper synth, ~2.5 s audio | ~1.6 s |
| Provider RSS after both models | ~430 MB |

On an idle machine and with a warm NIM connection, the plan's 3-6 s release
to first word is realistic. If it is not, check `logs/turns.jsonl` before
adding streaming.

## Tools and safety

File tools can only touch `workspace/`. Paths are resolved (including
symlinks) and must stay inside it.

`run_command`:

- One command only, parsed with `shlex.split`; no shell. Unquoted pipes,
  redirects, `&&`, `;` etc. are rejected.
- `sudo`, `su`, `doas`, `pkexec` are refused.
- These run without a prompt: `ls`, `cat` (short files), `head`, `tail`,
  `grep`, `wc`, `pwd`, `tree`, `echo`, `find` (without `-exec`, `-ok`,
  `-delete`, `-fprint`), plus exactly these git forms:
  - `git --no-pager status`
  - `git --no-pager diff --stat --no-ext-diff`
  - `git --no-pager log --oneline -20`
- Any argument that resolves outside `workspace/` asks first.
- **Anything that executes code always asks**, typed in the terminal, every
  time. There is no once-per-session approval.
- Commands run in their own process group with a 20 s timeout; the whole group
  is killed on timeout.

`godot_check` runs a fixed argv (`godot --headless --check-only --script ...`)
and does not prompt.

Git forms are only safe in repositories you trust: `.git/config` can configure
pagers, diff drivers or hooks that execute code.

Never log the API key. `.env` is gitignored.

## Configuration

Everything tunable is in `src/config.py`, with environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NVIDIA_API_KEY` | — | required |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | API endpoint |
| `NIM_MODEL` | `meta/llama-3.1-8b-instruct` | must pass the tool self-test |
| `WHISPER_MODEL` | `tiny.en` | try `base.en` if accuracy is poor |
| `WHISPER_CPU_THREADS` | `2` | CTranslate2 worker threads |
| `PTT_KEY` | `f9` | pynput key name |
| `PIPER_BIN` | auto | Piper executable (venv-local first) |
| `AUDIO_INPUT_DEVICE` | — | sounddevice index or name; empty = default |
| `AUDIO_OUTPUT_DEVICE` | auto | index/name; empty prefers pipewire/pulse |
| `DEBUG` | `0` | extra JSON-serializability check |

History caps: 12 turns, 40 messages, 60,000 characters. Tool output caps:
8 KB per result, 16 KB per turn. Recordings cap at 60 s. Directory listings
cap at 200 entries. Clips shorter than 0.3 s or below RMS 0.005 are ignored.

Trimming only cuts at user-message boundaries, so an assistant `tool_calls`
message is never separated from its `tool` replies.

## Logs

Each turn appends one JSON line to `logs/turns.jsonl`:

```json
{"ts":"...","transcript":"...","stt_ms":900,"llm_ms":2100,
 "tools":[{"name":"write_file","args":{},"ms":2}],
 "tts_ms":500,"tts_error":null,"error":null,"reply":"..."}
```

Use these numbers to decide whether Phase 3.5 streaming is worth it.

## Tests

```bash
pytest -q
```

- `test_tools.py` — path escapes, symlinks, truncation, command policy,
  process-group timeout, dispatcher errors.
- `test_agent.py` — fake LLM, round limit, bad tools/args/JSON, trim
  invariants and caps.
- `test_ptt.py` — key-repeat suppression and queue behavior.
- `test_audio_io.py` — clip guards and interruptible chunked playback.
- `test_stt_tts.py` — STT guards/wiring, markdown stripping, optional real
  Piper/Whisper smoke tests (skipped when models are absent).

The real speech test is opt-in:

```bash
VOICE_AGENT_STT_SMOKE=1 pytest tests/test_stt_tts.py -k fixture
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Whisper outputs gibberish / "Thank you." on silence | low mic gain, no silence guard | raise mic level, keep RMS guard, `condition_on_previous_text=False` |
| PTT key does nothing | Wayland, wrong key, permissions | use XFCE on X11; check `PTT_KEY`; try another key |
| Hang on key release | stream opened/stopped per press | keep the one persistent `InputStream` |
| Audio drops while transcribing | no native 16 kHz device | pick a native 16 kHz device or accept resampling |
| API 400 about tool messages | history trimmed mid tool group | trim only at user boundaries (already enforced) |
| Model never calls tools | model lacks tool calling | read startup warning, change `NIM_MODEL` |
| Piper attribute/signature errors | Piper API changed | use CLI path, pin `piper-tts` version |
| Godot processes linger | only direct child killed | process-group kill (already implemented) |
| Out of memory | browser + models | close apps, `tiny.en`, zram, thread caps |
| Slow replies | large model / long answers | smaller model, shorter prompt, lower `max_tokens` |

## Definition of done (v1)

Hold F9, say "create a GDScript file called player.gd with basic movement",
release, and within a few seconds hear a short confirmation while
`workspace/player.gd` exists with valid content. The app runs for 30 minutes
without crashing and stays under ~800 MB RAM.

## Optional later work

- **Phase 3.5 streaming** — `llm.chat_stream` + `tts.speak_stream` exist but are
  not wired in. Only build it if `logs/turns.jsonl` shows the gain is worth
  the added code. Never stream a turn that carries tools.
- **Wake word** — openWakeWord; needs the mic always on, extra CPU/RAM, and
  Silero VAD.
- **Godot tools** — search your RAG notes, run a scene, read the error log.
- **Offline fallback** — a small local Ollama model.
