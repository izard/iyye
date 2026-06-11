# Iyye
Iyye (home deity in Chuvash language) is an experimental always-on personal assistant architecture focused on persistent memory, and hopefully emergence of a distinct local personality that develops differently in each installation.
It is vibecoded Python project including lightweight "brain" loop, short-term and long-term memory, sensor/actuator IO, and multiple
cooperative processing streams.

The project is not very usable right now.

## What It Does
At runtime, `main_loop.py` creates an `IyyeBrain` and runs it once per second.
The brain:

- collects queued sensor input;
- keeps a global state machine: asleep, waking up, awake, winding down;
- runs processing streams in cooperative ticks;
- stores raw IO and stream activity logs on disk;
- extracts facts into short-term memory;
- promotes durable facts into long-term memory during sleep replay;
- uses local LLMs for chat, fact extraction, alignment scoring, stream codegen,
  and reflection when available.

## Project Goal: A Personal Assistant With a Growing Character
The central goal is to explore how a personal assistant can develop a distinct personality, habits, preferences, memories, and character traits over time.

Each installation is expected to become different. The assistant’s behavior should be shaped by its local environment, humans it interacts with, its memory history, available tools, recurring routines, and the feedback it receives. Two Iyye instances should not merely share the same codebase; they should gradually become different companions.

The project treats personality as an evolving system property rather than a static prompt. Streams, memory, sleep replay, self-reflection, alignment signals, and long-running interaction history all contribute to the assistant’s developing style of attention, judgment, priorities, and sense of continuity.

## Main Concepts
### Streams
Streams live in `streams/` and subclass `ProcessingStream` from `iyye_base.py`.
Each stream has input history, output history, activity log, current state,
priority, urgency, and alignment scores.

Important built-in streams include:

- `user_chat_stream`: handles local admin web chat and Telegram conversations.
- `attention_stream`: chooses which stream should be conscious/focused.
- `alignment_stream`: scores streams against the motivation goals.
- `stm_update_stream`: extracts facts from stream logs into short-term memory.
- `self_reflection_stream`: monitors system state and suggests new streams.
- `stream_factory`: creates new processing streams from suggestions or gaps.
- `adenosine_stream`: tracks tiredness/energy and triggers wind-down.
- `llm_management_stream`: starts, stops, and monitors local LLMs.
- `theory_of_mind_stream`: tracks contact-specific social context.

### Memory
Short-term memory is stored in `stm_history/` as JSONL facts. Each fact includes
confidence, provenance, time frame, timestamp, and optional media.

Long-term memory is handled by `iyye_io/memory_mcp_client.py` and backed by
LanceDB plus embeddings. Sleep replay promotes durable STM facts into LTM and
filters out transient system noise.

### IO
Sensors and actuators live in `iyye_io/`.

Implemented or partially implemented:

- Local web chat sensor and actuator.
- Telegram sensor and actuator, when shell environment credentials are set.
- Hardware sensor for CPU, memory, disk, and optional GPU metrics.
- Git MCP sensor for source/prompt changes and controlled writes.
- Memory MCP client.

Camera, microphone, STT, and TTS paths exist as stubs or partial experiments and
should not be treated as working core functionality yet.

## Setup
Use the project virtual environment before running commands:

```bash
source .venv/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Telegram requires credentials in the shell environment:

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_DEFAULT_CHAT_ID="..."
```

Without those values, Telegram IO is disabled or unable to send.

## Running
Start the brain and local web UI:

```bash
source .venv/bin/activate
python main_loop.py
```

Then open:

```text
http://127.0.0.1:5000/chat
```

For a short development run:

```bash
python main_loop.py 60
```

For verbose logs:

```bash
python main_loop.py --debug
```

Local LLMs are managed by scripts in `tools/`. The active model registry is in
`tools/llm-registry.json`; status helpers include:

```bash
tools/llm-status-all.sh
tools/llm-stop.sh
```

## Generated Streams
The factory can ask an LLM to generate new `ProcessingStream` subclasses. This
is one of the most experimental parts of the project.

Known issue: automatically generated `llm_suggested_*` streams have often been
low-value. Older generated streams mostly did not consume the sensor data that
caused their creation; instead they searched long-term memory with generic
queries such as "interesting fact", "concept", or "?". That leads to vague
observations, repetitive proactive messages, and STM facts that are later
filtered as operational noise.

## Project Layout
```text
main_loop.py              brain loop and state machine
iyye_base.py                base sensor, actuator, and stream classes
iyye_hld.md                 high-level design document
llm_client.py               OpenAI-compatible local LLM wrapper
mcp_client.py               MCP stdio client helpers
web_chat_2.py               simple Flask web chat UI
iyye_io/                   sensors, actuators, memory, MCP servers
streams/                   processing streams
prompts/                   LLM prompt templates
tools/                     local LLM scripts and status helpers
tests/                     small test experiments
```

Runtime data directories are created as needed:

```text
io_history/                raw sensor history
streams_history/           stream activity logs
stm_history/               short-term memory facts and media
memory_db/ or memory_db.*   long-term memory storage
```

## Development Notes
- Always activate `.venv` before running Python commands.
- The worktree may contain generated stream files and generated deletions
- Direct API use still exists in older built-in IO. MCP is more important for
  newly created IO/stream integrations than for all legacy/built-in modules.
- Sleep behaves differently on first process start versus after a completed day:
  first sleep skips replay, later sleep cycles consolidate the last awake log.
- The full `pytest` command may collect vendored `tools/llama.cpp` tests. Prefer
  targeted tests such as `pytest tests` unless you intentionally want to test
  vendored tooling.

## Current Status
Iyye can run, chat locally, keep memory, collect hardware/git/Telegram input
under the right environment, and exercise its sleep/wake stream architecture.

It is not yet a useful and/or polished agent.
