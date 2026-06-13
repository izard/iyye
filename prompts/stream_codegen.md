---
description: Generate a new ProcessingStream subclass to pursue a specific alignment goal
variables:
  source_name: Name of the source stream that triggered this generation
  primary_goal: Primary alignment goal to pursue (curiosity, agency, social, self_preservation)
  alignment_scores: Alignment scores for the source stream, comma-separated goal=score pairs
  recent_inputs: Last few inputs from the source stream, one per line prefixed with -
  recent_outputs: Last few outputs from the source stream, one per line prefixed with -
  class_name: Exact Python class name to use
  file_name: Filename this stream will be saved as (for reference only)
  sensor_key: Exact sensor queue key in context['sensors_data'] (or "(none)" for goal-driven streams)
---

You are generating a Python processing stream for Iyye, an AI personality.

## Task

Write a class named `{class_name}` that subclasses `ProcessingStream` (imported from `iyye_base`).

This stream was created by `{source_name}` to pursue the goal: **{primary_goal}**

**Sensor queue key:** `{sensor_key}`
If the sensor key is `(none)`, this is a **goal-driven stream** with no sensor input. Do NOT try to read from `sensors_data` — there is nothing there for you. Instead, use only `context['memory']`, `context['stm']`, `context['adenosine']`, and `context['streams']` as your inputs. Do NOT send proactive greetings or "checking in" messages — only act when you find something genuinely interesting in memory or state.
If the sensor key is a real name (not `(none)`), read its data with exactly: `sensors_data.get('{sensor_key}', [])` — do NOT guess or modify the sensor name.

Alignment context: {alignment_scores}

## Source stream recent inputs
{recent_inputs}

## Source stream recent outputs
{recent_outputs}

## Base class interface

```python
from iyye_base import ProcessingStream

class ProcessingStream:
    name: str
    priority: int           # 1-10
    urgency: float          # 0.0-1.0
    alignment_scores: dict
    is_conscious: bool
    input_history: list
    output_history: list
    activity_log: list

    def add_to_log(self, msg: str): ...    # record an action in the activity log
    def checkpoint(self) -> int: ...       # cooperative yield point (call regularly)
    def add_input(self, data, source=""): ...
    def add_output(self, data, target=""): ...
```

The `execute(self, context)` method receives:
- `context['sensors_data']` — dict of sensor_name -> list of raw messages
- `context['memory']`       — long-term memory (read-only during awake); `.search(query)` returns a **list of dicts** with keys `text`, `confidence`, `source`, etc. — extract the `.get('text', '')` field from each result before using it. `.search_text(query)` works the same way. **NEVER embed raw search results in user-facing messages** — always extract and summarize the `text` fields
- `context['stm']`          — short-term memory; `.add_fact(text, confidence, provenance, time_frame, media_path=None)` to store discovered facts (promoted to LTM during sleep); `.search(query, limit=10)` for substring search (returns list of fact dicts, newest first); `.get_recent(limit=20)` for the latest facts. To attach supporting media (images, audio), first call `.save_media(data_bytes, filename)` which returns a tgz path, then pass it as `media_path`
- `context['adenosine']`    — float 0-1, current energy level (1.0 = fully rested, drains toward 0; sleep triggers at ~0.15)
- `context['actuators']`    — dict of actuator_name -> actuator with `.actuate(text)`. Only use `"web_chat_actuator"` (local web UI). **NEVER send to TelegramActuator or tts_actuator** — external channels are managed by dedicated trusted streams only
- `context['streams']`      — list of all currently active streams

## Rules

1. `__init__(self)` must take no arguments and call `super().__init__(name="...")` with a short descriptive name — this allows the stream to be reloaded automatically on restart
2. `execute(self, context)` must call `self.checkpoint()` at least once and return a dict
3. Wrap all sensor/actuator/memory access in try/except
4. Only import from: `os`, `re`, `json`, `logging`, `datetime`, `typing`, `iyye_base`
5. **NEVER override** `checkpoint`, `add_to_log`, `add_input`, or `add_output` — the base class implementations handle cooperative stop, persistent logging, and timestamped history. Just call them as-is.
6. **NEVER set** `self.is_conscious = True` in `__init__` — consciousness is assigned dynamically by the attention stream at runtime
7. **NEVER send raw data structures** (dicts, lists, JSON) to actuators — messages to users must be short, natural-language sentences. Extract `.get('text', '')` from memory results before composing a message

## Goal guidance

- **curiosity**: search `context['memory']` for something new or connect existing facts; store a discovered fact via `context['stm'].add_fact(text, confidence, provenance, time_frame)`. Do NOT message the user unless you found a genuinely novel connection
- **agency**: compose a message and send via `web_chat_actuator` only — but only when there is something specific and useful to say. NEVER send generic "checking in" or "I'm here" messages
- **social**: look for unanswered questions or unfinished topics in memory. Only send a follow-up if you find a concrete topic to reference. NEVER send generic greetings or "just checking in" messages
- **self_preservation**: inspect `context['adenosine']` and sensor/stream health; log concerns to activity log only — do NOT message the user about routine system metrics

Do NOT set `_factory_created = True` — the stream must be reloadable on restart.
Write only the Python code, inside a single ```python ... ``` block. No text outside the block.
