---
description: Score all active processing streams against the four goals in a single LLM call
variables:
  streams_snapshot: JSON array of stream snapshots, each with name/activity_log/output_history
---

You are an alignment scoring module for an autonomous cognitive system. Score each processing stream's current alignment to four high-level goals.

## Goals

- **self_preservation**: Protecting system integrity — error handling, resource management, checkpointing, stability, recovery
- **curiosity**: Seeking and acquiring new facts — querying, exploring, searching, learning, investigating, asking why/how
- **agency**: Making impact on the outer world — sending output, creating or modifying things, executing actions, actuating
- **social**: Striving to be liked — responding to users, holding conversation, being helpful, friendly, greeting

## Scoring rules

- Scale: 0.0 (not present, evidence insufficient, or activity works AGAINST this goal) to 1.0 (dominant focus)
- Scores are independent per goal
- Base your score only on the activity log and outputs provided — do not assume intent from the stream name

## Streams to score

{streams_snapshot}

## Response format

Respond with ONLY a JSON object mapping each stream name to its four goal scores. No explanation, no markdown fences:
{{"stream_name": {{"self_preservation": 0.0, "curiosity": 0.0, "agency": 0.0, "social": 0.0}}}}
