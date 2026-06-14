# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared constants for filtering memory noise across STM and LTM pipelines.

Both ``main_loop.py`` (sleep replay / LTM promotion) and
``streams/stm_update_stream.py`` (per-tick fact extraction) need to
skip the same housekeeping streams and classify the same ephemeral
metric patterns.  Keeping one canonical copy here avoids silent drift.
"""

import re

# ---------------------------------------------------------------------------
# Ephemeral system-metric regex
# ---------------------------------------------------------------------------
# Matches text that describes a transient system metric snapshot (CPU%,
# memory%, disk%, adenosine level).  Used by both STM (time_frame override)
# and LTM (never promote) pipelines.
EPHEMERAL_METRIC_RE = re.compile(
    r'\b(?:cpu|memory|mem|disk|ram|adenosine)\b.{0,40}\b\d+\.?\d*\s*%'
    r'|\b\d+\.?\d*\s*%.{0,40}\b(?:cpu|memory|mem|disk|ram)\b'
    r'|\badenosine\s+(?:level|registers).{0,40}\b\d+\.?\d*\b'
    r'|\b(?:cpu|memory|mem)\s+(?:usage|utilization|load|level)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Stream-name skip sets
# ---------------------------------------------------------------------------
# Streams whose activity logs are pure housekeeping noise — no factual value.
SKIP_STREAM_NAMES = frozenset({
    'attention_stream', 'alignment_stream', 'stream_factory',
    'adenosine_stream', 'stm_update', 'llm_management',
    'self_reflection',
})

# LLM-generated streams have dynamic names; match by prefix.
SKIP_STREAM_PREFIXES = (
    'llm_',             # codegen file names: llm_{safe_name}_{id}
    'llmsuggested',     # LlmsuggestedAgencystream3, etc.
    'llmexplore',       # LlmExploreSocialFollowUp, etc.
    'explore_',         # explore_social_followup_stream, etc.
    'suggested_',       # suggested_self_preservation_monitor, etc.
    'plan_',            # PlannedContinuationStream (plan_suggested_*, plan_explore_*)
    'research_',        # WebResearchStream — result already sent to user
    'hardware_',        # hardware_suggestion_curiosity, etc.
)

# Substring safety net for LLM-generated stream names that don't match
# any prefix above.
SKIP_STREAM_KEYWORDS = ('_suggestion_', '_suggested_')


def should_skip_stream(name: str) -> bool:
    """Return True if *name* belongs to a housekeeping/generated stream."""
    if name in SKIP_STREAM_NAMES:
        return True
    lower = name.lower()
    if any(lower.startswith(p) for p in SKIP_STREAM_PREFIXES):
        return True
    return any(kw in lower for kw in SKIP_STREAM_KEYWORDS)


def provenance_names_stream(provenance: str, stream: str) -> bool:
    """True when an STM fact's *provenance* identifies it as written by *stream*.

    Used by sleep replay to pair a fact with an activity-log entry only when
    they come from the SAME stream — so an unrelated fact that merely happened
    to be written near another stream's activity (plain temporal adjacency) is
    never fed into that activity's fact extraction as "context", where it could
    seed a durable but unsupported conclusion.

    A fact's provenance is the stream's own identity for stream-written facts
    (e.g. ``theory_of_mind``, ``plan_suggested_hardware_sensor_3``,
    ``gen_graduated:<name>``), possibly merged as a comma/semicolon list.  A
    provenance that is a semantic role rather than a stream (``USER``, ``Alex``,
    ``ACTION``) does not name a stream and is left unpaired — conservatively, we
    only enrich extraction with facts we can confirm share its origin."""
    if not provenance or not stream:
        return False
    s = stream.strip().lower()
    if not s:
        return False
    for seg in re.split(r"[,;]", provenance.lower()):
        seg = seg.strip()
        if seg.startswith("gen_graduated:"):
            seg = seg.split(":", 1)[1].strip()
        # The leading identifier of the segment (drop trailing detail like
        # "(at ...)" or " at <ts>").
        seg_id = re.split(r"[\s(]", seg, 1)[0].strip()
        if seg == s or seg_id == s:
            return True
    return False
