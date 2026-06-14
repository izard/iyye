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
"""Versioned prompts with outcome-based selection — self-improvement for the
LLM prompts (gap #6: prompts were static files, never tuned from outcomes).

The shipped ``prompts/<name>.md`` is the implicit baseline ("base").  A sleep
pass can propose an improved version (LLM rewrite); the registry trials it as
the active version, attributes outcomes to whichever version was serving, and
**conservatively** keeps the challenger only if it beats the baseline by a
margin over a minimum sample — otherwise it rolls back.  Default behaviour is
unchanged: with no registered versions, prompt loading falls through to the
shipped file.

Persistence: ``prompt_versions/registry.json`` (metadata + per-version stats)
and ``prompt_versions/<name>/<vid>.md`` (candidate content).  Atomic writes.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Prompts")

# A challenger must accumulate at least this many outcomes before it can be
# judged, and beat the baseline mean by this margin to be promoted — so a
# prompt rewrite is adopted only on real, sustained improvement.
_MIN_SAMPLES = 20
_PROMOTE_MARGIN = 0.02


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class PromptRegistry:
    def __init__(self, base_dir: Optional[Path] = None):
        self._dir = Path(base_dir) if base_dir else (PROJECT_ROOT / "prompt_versions")
        self._registry_path = self._dir / "registry.json"
        self._lock = threading.RLock()
        # name -> {"active": vid, "versions": {vid: {...}}, "trial": {...}|None}
        self._reg: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._registry_path.exists():
                data = json.loads(self._registry_path.read_text())
                if isinstance(data, dict):
                    self._reg = data
        except Exception as exc:
            log.warning("PromptRegistry: could not load registry: %s", exc)

    def _save(self) -> bool:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            tmp = self._registry_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._reg, indent=2))
            tmp.replace(self._registry_path)
            return True
        except Exception as exc:
            log.warning("PromptRegistry: could not save registry: %s", exc)
            return False

    def _version_path(self, name: str, vid: str) -> Path:
        return self._dir / name / f"{vid}.md"

    # ------------------------------------------------------------------
    # Resolution (used by the prompt loader)
    # ------------------------------------------------------------------

    def active_version_id(self, name: str) -> str:
        with self._lock:
            rec = self._reg.get(name)
            return rec.get("active", "base") if rec else "base"

    def active_content(self, name: str) -> Optional[str]:
        """The active version's content, or None when the baseline (shipped
        file) is active — the caller then loads the file (default behaviour)."""
        with self._lock:
            rec = self._reg.get(name)
            if not rec:
                return None
            vid = rec.get("active", "base")
            if vid == "base":
                return None
            try:
                return self._version_path(name, vid).read_text(encoding="utf-8")
            except Exception as exc:
                log.warning("PromptRegistry: active version %s/%s unreadable "
                            "(%s) — falling back to base", name, vid, exc)
                return None

    # ------------------------------------------------------------------
    # Outcome attribution (sleep pass folds outcomes per active version)
    # ------------------------------------------------------------------

    def ensure_tracked(self, name: str) -> None:
        """Begin tracking *name* with an implicit "base" version so the baseline
        accrues an outcome history (the bar a future challenger must clear),
        without changing what serves traffic — base still resolves to the file."""
        with self._lock:
            if name not in self._reg:
                self._reg[name] = {
                    "active": "base",
                    "versions": {"base": {"status": "active", "parent": None,
                                          "n": 0, "sum_reward": 0.0,
                                          "created": _utcnow()}},
                    "trial": None,
                }
                self._save()

    def record_outcome(self, name: str, reward: float,
                       version_id: Optional[str] = None) -> None:
        """Credit a version with one outcome's reward.

        *version_id* names the version that produced the outcome — the sleep
        fold passes the value journaled on the ``llm_submit`` so attribution is
        exact across a mid-cycle swap.  When omitted, the currently-active
        version is credited."""
        with self._lock:
            rec = self._reg.get(name)
            if not rec:
                return
            vid = version_id or rec["active"]
            v = rec["versions"].get(vid)
            if v is not None:
                v["n"] = v.get("n", 0) + 1
                v["sum_reward"] = v.get("sum_reward", 0.0) + float(reward)
                self._save()

    # ------------------------------------------------------------------
    # Trialling a challenger (proposed by the sleep improvement pass)
    # ------------------------------------------------------------------

    def start_trial(self, name: str, candidate_content: str) -> Optional[str]:
        """Register *candidate_content* as a new version and make it active for
        a trial, recording the baseline it must beat.  Returns the version id,
        or None if a trial is already running for this prompt."""
        with self._lock:
            rec = self._reg.setdefault(name, {
                "active": "base",
                "versions": {"base": {"status": "active", "parent": None,
                                      "n": 0, "sum_reward": 0.0,
                                      "created": _utcnow()}},
                "trial": None,
            })
            if rec.get("trial"):
                return None  # one trial at a time
            baseline = rec["active"]
            bstats = rec["versions"].get(baseline, {})
            baseline_mean = (bstats.get("sum_reward", 0.0) / bstats["n"]
                             if bstats.get("n") else 0.0)
            vid = f"v{len([k for k in rec['versions'] if k != 'base']) + 1}"
            try:
                p = self._version_path(name, vid)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(candidate_content, encoding="utf-8")
            except Exception as exc:
                log.warning("PromptRegistry: could not write %s/%s: %s",
                            name, vid, exc)
                return None
            rec["versions"][vid] = {"status": "trial", "parent": baseline,
                                    "n": 0, "sum_reward": 0.0,
                                    "created": _utcnow()}
            rec["active"] = vid
            rec["trial"] = {"candidate": vid, "baseline": baseline,
                            "baseline_mean": round(baseline_mean, 4)}
            self._save()
            return vid

    def select(self, name: str) -> Optional[str]:
        """Evaluate a running trial: promote the challenger if it beats the
        baseline by the margin over enough samples, roll back if it clearly
        does not.  Returns a short decision string, or None (still trialling)."""
        with self._lock:
            rec = self._reg.get(name)
            trial = rec.get("trial") if rec else None
            if not trial:
                return None
            cand = trial["candidate"]
            cv = rec["versions"].get(cand, {})
            n = cv.get("n", 0)
            if n < _MIN_SAMPLES:
                return None  # not enough trial data yet
            cmean = cv.get("sum_reward", 0.0) / n
            if cmean >= trial["baseline_mean"] + _PROMOTE_MARGIN:
                rec["versions"][cand]["status"] = "active"
                old = trial["baseline"]
                if old in rec["versions"] and old != cand:
                    rec["versions"][old]["status"] = "retired"
                rec["active"] = cand
                rec["trial"] = None
                self._save()
                return f"promoted {cand} (mean {cmean:.3f} > base {trial['baseline_mean']:.3f})"
            if n >= 2 * _MIN_SAMPLES:
                # Enough evidence the challenger isn't better — roll back.
                rec["active"] = trial["baseline"]
                rec["versions"][cand]["status"] = "retired"
                rec["trial"] = None
                self._save()
                return f"rolled back {cand} (mean {cmean:.3f} <= base {trial['baseline_mean']:.3f})"
            return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self, name: str) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._reg.get(name, {})))

    def tracked_names(self) -> List[str]:
        with self._lock:
            return list(self._reg.keys())


def fold_outcomes(reg: PromptRegistry,
                  events: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Attribute a cycle's LLM outcomes to the prompt version that served each
    job, accumulating per-version reward in *reg*.  Pure over the journal: a
    job's reward is 1.0 when its result landed ``ok`` and was not discarded,
    else 0.0 — a prompt-agnostic success signal.  Returns per-name aggregate
    counts ``{name: {"n", "sum"}}`` for journaling/introspection."""
    served: Dict[str, Any] = {}  # job_id -> (name, version)
    for e in events:
        if e.get("type") == "llm_submit" and e.get("prompt_name"):
            served[e.get("job_id")] = (e["prompt_name"],
                                       e.get("prompt_version", "base"))
    counts: Dict[str, Dict[str, float]] = {}
    tracked = set()
    for e in events:
        if e.get("type") != "llm_result" or e.get("discarded"):
            continue
        ref = served.get(e.get("job_id"))
        if not ref:
            continue
        name, version = ref
        reward = 1.0 if e.get("ok") else 0.0
        if name not in tracked:
            reg.ensure_tracked(name)
            tracked.add(name)
        reg.record_outcome(name, reward, version_id=version)
        c = counts.setdefault(name, {"n": 0.0, "sum": 0.0})
        c["n"] += 1
        c["sum"] += reward
    return counts


# A prompt is eligible for a rewrite trial once its baseline has enough
# evidence and visible failure headroom — don't churn an already-reliable one.
_PROMPT_MIN_BASELINE = 30
_PROMPT_IMPROVE_CEILING = 0.95


def _placeholders(text: str) -> set:
    """The ``{name}`` substitution slots a prompt template exposes."""
    return set(re.findall(r"{(\w+)}", text or ""))


def select_prompt_to_improve(
    reg: PromptRegistry, names: List[str],
    min_baseline: int = _PROMPT_MIN_BASELINE,
    ceiling: float = _PROMPT_IMPROVE_CEILING,
) -> Optional[str]:
    """Pick the worst-performing prompt worth rewriting, or None.

    Eligible = currently on its baseline ("base") with no trial running, with
    at least *min_baseline* recorded outcomes and a mean reward below *ceiling*
    (failure headroom).  Among those, the lowest-mean prompt has the most ROI."""
    best: Optional[str] = None
    best_mean = ceiling
    for name in names:
        rec = reg.status(name)
        if not rec or rec.get("active") != "base" or rec.get("trial"):
            continue
        base = rec.get("versions", {}).get("base", {})
        n = base.get("n", 0)
        if n < min_baseline:
            continue
        mean = base.get("sum_reward", 0.0) / n
        if mean < best_mean:
            best, best_mean = name, mean
    return best


def validate_candidate(base: str, candidate: str) -> Tuple[bool, str]:
    """Gate an LLM-proposed rewrite before it can serve traffic.

    Rejects an empty/identical rewrite, one that changes the placeholder set
    (``.format`` would break or silently drop context), or one whose length is
    implausible (truncated or runaway).  Returns ``(ok, reason)``."""
    cand = (candidate or "").strip()
    b = (base or "").strip()
    if not cand:
        return False, "empty"
    if cand == b:
        return False, "identical to base"
    if _placeholders(cand) != _placeholders(b):
        return False, "placeholder set changed"
    if not (0.4 * len(b) <= len(cand) <= 3.0 * len(b)):
        return False, "implausible length"
    return True, "ok"


# Process-wide singleton so the prompt loader and the sleep pass share state.
_REGISTRY: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PromptRegistry()
    return _REGISTRY


__all__ = ["PromptRegistry", "get_registry", "fold_outcomes",
           "select_prompt_to_improve", "validate_candidate"]
