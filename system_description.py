# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Canonical builder and renderer for ``system_description.md``.

HLD says both the brain (at sleep system-check) and SelfReflectionStream
(during awake gather ticks) maintain this file.  Two writers producing
two different formats — and UserChatStream preferring the in-memory copy
over the on-disk copy — meant the chat prompt context could show stale
self-reflection output from the previous awake cycle.

This module is the single source of truth for what the file looks like.
Both writers call :func:`publish_system_description` with a canonical
state dict; rendering, cache update, and (deduplicated) disk write all
happen in one place.

Canonical state schema (all keys optional — missing values render
gracefully):

    {
        'timestamp':        str,        # ISO 8601 UTC; defaults to "now"
        'iyye_day':         int | None, # shown in title if present
        'hardware': {
            'cpu_percent':    float,
            'memory_percent': float,
            'disk_percent':   float,
        },
        'sensors': [                    # may be empty
            {'name': str, 'queue_size': int, 'healthy': bool | None}, ...
        ],
        'actuators': [                  # may be empty
            {'name': str, 'reachable': bool | None}, ...
        ],
        'llms':                          # None → "unavailable" / [] → "unreachable"
            None | [
                {'name': str, 'healthy': bool, 'size_gb': float,
                 'roles': [str]}, ...
            ],
        'streams': [                    # may be empty
            {'name': str, 'priority': int, 'is_conscious': bool,
             'pending': int | None}, ...
        ],
        'conscious_stream': str | None,
        'memory_facts':     int,
        'adenosine':        float,
        'adenosine_max':    float,
    }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.SystemDescription")

_MD_PATH = PROJECT_ROOT / "system_description.md"


def render_system_description(state: Dict[str, Any]) -> str:
    """Render canonical state into the markdown shown to chat streams."""
    ts = state.get('timestamp') or datetime.now(timezone.utc).isoformat()
    iyye_day = state.get('iyye_day')
    title_suffix = f" — Iyye day {iyye_day}" if iyye_day is not None else ""

    hw = state.get('hardware') or {}
    hw_lines = (
        f"| CPU      | {hw.get('cpu_percent', 0):.1f}% |\n"
        f"| Memory   | {hw.get('memory_percent', 0):.1f}% |\n"
        f"| Disk     | {hw.get('disk_percent', 0):.1f}% |"
    )

    sensors: List[Dict[str, Any]] = state.get('sensors') or []
    if sensors:
        sensor_rows = []
        for s in sensors:
            row = f"- **{s.get('name', '?')}**: queue={s.get('queue_size', 0)}"
            if s.get('healthy') is not None:
                row += f", healthy={s['healthy']}"
            sensor_rows.append(row)
        sensors_md = "\n".join(sensor_rows)
    else:
        sensors_md = "_(none)_"

    actuators: List[Dict[str, Any]] = state.get('actuators') or []
    if actuators:
        actuator_rows = []
        for a in actuators:
            row = f"- **{a.get('name', '?')}**"
            if a.get('reachable') is not None:
                row += f": reachable={a['reachable']}"
            actuator_rows.append(row)
        actuators_md = "\n".join(actuator_rows)
    else:
        actuators_md = "_(none)_"

    llms = state.get('llms')
    if llms is None:
        llm_md = "_(unavailable)_"
    elif not llms:
        llm_md = "_(unreachable)_"
    else:
        llm_md = "\n".join(
            f"- **{m.get('name', '?')}** "
            f"[{'UP' if m.get('healthy') else 'DOWN'}] "
            f"{m.get('size_gb', 0)}GB "
            f"roles=[{', '.join(m.get('roles', []) or []) or '-'}]"
            for m in llms
        )

    streams: List[Dict[str, Any]] = state.get('streams') or []
    conscious_name = state.get('conscious_stream') or "_(none)_"
    if streams:
        stream_rows = []
        for s in streams:
            row = (f"- **{s.get('name', '?')}** "
                   f"(priority={s.get('priority', 0)}")
            if s.get('pending') is not None:
                row += f", pending={s['pending']}"
            if s.get('is_conscious'):
                row += ", conscious"
            row += ")"
            stream_rows.append(row)
        streams_md = "\n".join(stream_rows)
    else:
        streams_md = "_(none)_"

    facts_count = state.get('memory_facts', 0)
    adenosine = state.get('adenosine', 0.0)
    adenosine_max = state.get('adenosine_max', 1.0)

    plan_lines: List[str] = state.get('plans') or []
    plans_md = "\n".join(f"- {line}" for line in plan_lines) or "_(none)_"

    return (
        f"# Iyye System Description\n"
        f"_Generated: {ts} UTC{title_suffix}_\n\n"
        f"## Hardware\n"
        f"| Resource | Usage |\n"
        f"|----------|-------|\n"
        f"{hw_lines}\n\n"
        f"## Sensors\n{sensors_md}\n\n"
        f"## Actuators\n{actuators_md}\n\n"
        f"## LLMs\n{llm_md}\n\n"
        f"## Execution Streams\n"
        f"Active: {len(streams)}  |  Conscious: {conscious_name}\n\n"
        f"{streams_md}\n\n"
        f"## Long-term Memory\n"
        f"Stored facts: {facts_count}\n\n"
        f"## Long-term Plans\n{plans_md}\n\n"
        f"## Adenosine\n"
        f"Level: {adenosine:.3f} / {adenosine_max:.1f}\n"
    )


def publish_system_description(brain: Any, state: Dict[str, Any]) -> str:
    """Render *state* and publish to brain cache + disk (if content changed).

    Both ``brain._system_description_md`` (read by UserChatStream every
    tick) and the on-disk file are updated by this single path so neither
    can fall behind the other.  Disk write is skipped when the rendered
    content is byte-identical to what was last persisted — avoids spamming
    inotify/file-watcher tools on quiet days.

    Returns the rendered markdown so callers can log or further process it.
    """
    # Long term plans come straight from the shared store so both writers
    # (awake self-reflection, brain sleep check) show them without each
    # having to assemble the section.
    if 'plans' not in state:
        store = getattr(brain, 'plan_store', None)
        if store is not None:
            try:
                state = dict(state)
                state['plans'] = [p.summary_line() for p in store.all_plans()]
            except Exception as exc:
                log.warning("Could not include plans in description: %s", exc)
    md = render_system_description(state)
    try:
        brain._system_description_md = md
    except Exception as exc:
        log.warning("Could not update brain cache: %s", exc)

    last_disk = getattr(brain, '_system_description_disk', None)
    if md == last_disk:
        return md
    try:
        _MD_PATH.write_text(md, encoding="utf-8")
        try:
            brain._system_description_disk = md
        except Exception:
            pass
        log.info("Updated %s (%d chars)", _MD_PATH.name, len(md))
    except Exception as exc:
        log.warning("Failed to write %s: %s", _MD_PATH.name, exc)
    return md


def _load_llm_status(brain: Any) -> Optional[List[Dict[str, Any]]]:
    """Read tools/llm-active.json for the LLM section, or None if unavailable."""
    path = PROJECT_ROOT / "tools" / "llm-active.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not read llm-active.json: %s", exc)
        return None
    return [
        {'name': m.get('name', '?'), 'healthy': bool(m.get('healthy', False)),
         'size_gb': m.get('size_gb', 0), 'roles': m.get('roles', []) or []}
        for m in doc.get('models', [])
    ]


def run_system_check(brain: Any) -> Dict[str, Any]:
    """Inspect the running system and (re)publish ``system_description.md``.

    HLD: the asleep system-check examines "inputs, actuators, memory, hardware
    … etc." and writes the markdown description used by awake streams.  This is
    the single owner of *producing* that description; both SelfReflectionStream
    (which owns keeping it current, per HLD) and the brain's first-sleep
    bootstrap call it.  Returns the raw state dict for logging/inspection.
    """
    import psutil

    current_conscious_name = getattr(
        getattr(brain, '_current_conscious', None), 'name', None
    )
    sensors = getattr(brain, 'sensors', {}) or {}
    actuators = getattr(brain, 'actuators', {}) or {}
    streams = getattr(brain, 'streams', []) or []
    state = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'sensors': {
            name: {'queue_size': len(q),
                   'maxlen': q.maxlen if hasattr(q, 'maxlen') else None}
            for name, q in sensors.items()
        },
        'actuators': list(actuators.keys()),
        'memory_facts': brain.memory.count(),
        'active_streams': len(streams),
        'conscious_stream': current_conscious_name,
        'hardware': {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
        },
        'adenosine': brain.adenosine.level,
    }
    log.info("System state check: %d sensors, %d actuators, %d streams, "
             "CPU=%.1f%%, Mem=%.1f%%",
             len(sensors), len(actuators), len(streams),
             state['hardware']['cpu_percent'], state['hardware']['memory_percent'])

    canonical = {
        'timestamp':        state['timestamp'],
        'iyye_day':         getattr(brain, 'iyye_day', None),
        'hardware':         state['hardware'],
        'sensors':          [
            {'name': name, 'queue_size': info.get('queue_size', 0), 'healthy': None}
            for name, info in state['sensors'].items()
        ],
        'actuators':        [{'name': n, 'reachable': None} for n in state['actuators']],
        'llms':             _load_llm_status(brain),
        'streams':          [
            {'name': s.name, 'priority': s.priority,
             'is_conscious': s.name == current_conscious_name, 'pending': None}
            for s in streams
        ],
        'conscious_stream': current_conscious_name,
        'memory_facts':     state['memory_facts'],
        'adenosine':        state['adenosine'],
        'adenosine_max':    brain.adenosine.MAX,
    }
    publish_system_description(brain, canonical)
    return state


__all__ = [
    'render_system_description',
    'publish_system_description',
    'run_system_check',
]
