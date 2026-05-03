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
    'llmsuggested',     # LlmsuggestedAgencystream3, etc.
    'llm_suggested',    # llm_suggested_hardware_stream, etc.
    'llmexplore',       # LlmExploreSocialFollowUp, etc.
    'explore_',         # explore_social_followup_stream, etc.
    'suggested_',       # suggested_self_preservation_monitor, etc.
    'plan_suggested',   # PlannedContinuationStream fallback streams
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
