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
# streams/llm_management_stream.py
#!/usr/bin/env python3
"""
LLM Management subconscious Stream

Also hosts LLMRouter — the single routing layer through which all LLM
requests flow.  Every call is tracked (model, role, latency, token counts)
and a live summary is written to tools/llm-active.json each health tick so
the hardware sensor and self-reflection can report on LLM utilisation.

Optional env vars:
  LLM_HOST             — defaults to 127.0.0.1
  LLM_AUTO_START       — if set, auto-start the default chat model when unhealthy
  LLM_DEFAULT_MODEL    — script name under tools/ (default: from registry default_for chat)
  LLM_VISION_AUTO_START — if set, also auto-start the default vision model
"""

import collections
import json
import logging
import os
import subprocess
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

from iyye_base import ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")

_TOOLS_DIR   = Path(__file__).parent.parent / "tools"
_REGISTRY_PATH = _TOOLS_DIR / "llm-registry.json"
_ACTIVE_PATH   = _TOOLS_DIR / "llm-active.json"


# ======================================================================
# Proxy wrappers — intercept LLM calls and record stats via LLMRouter
# ======================================================================

class _LLMClientProxy:
    """Wraps LLMClient; forwards all calls and records stats."""

    def __init__(self, client, model_name: str, role: str, router: "LLMRouter") -> None:
        self._client     = client
        self._model_name = model_name
        self._role       = role
        self._router     = router

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        t0 = time.monotonic()
        result = self._client.complete(user_prompt, system_prompt)
        self._router._record(
            self._model_name, self._role,
            len(user_prompt), len(result), time.monotonic() - t0,
        )
        return result

    def complete_from_file(self, prompt_name: str, **variables) -> str:
        t0 = time.monotonic()
        result = self._client.complete_from_file(prompt_name, **variables)
        prompt_chars = sum(len(str(v)) for v in variables.values())
        self._router._record(
            self._model_name, self._role,
            prompt_chars, len(result), time.monotonic() - t0,
        )
        return result

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class _VisionClientProxy:
    """Wraps VisionClient; forwards all calls and records stats."""

    def __init__(self, client, model_name: str, role: str, router: "LLMRouter") -> None:
        self._client     = client
        self._model_name = model_name
        self._role       = role
        self._router     = router

    def describe_image_bytes(self, image_bytes: bytes, **kwargs) -> str:
        t0 = time.monotonic()
        result = self._client.describe_image_bytes(image_bytes, **kwargs)
        self._router._record(
            self._model_name, self._role,
            len(image_bytes), len(result), time.monotonic() - t0,
        )
        return result

    def describe_image_file(self, image_path, **kwargs) -> str:
        t0 = time.monotonic()
        result = self._client.describe_image_file(image_path, **kwargs)
        self._router._record(
            self._model_name, self._role,
            0, len(result), time.monotonic() - t0,
        )
        return result

    def __getattr__(self, name: str):
        return getattr(self._client, name)


# ======================================================================
# LLMRouter
# ======================================================================

class LLMRouter:
    """
    Central router for all LLM requests.

    Usage (from any stream that has brain access):
        llm  = brain.llm_router.get_client(role="chat", no_think=True)
        vis  = brain.llm_router.get_client(role="vision")

    `get_client` returns a proxy that:
    - Forwards complete() / complete_from_file() / describe_image_bytes() to
      the underlying LLMClient or VisionClient.
    - Records every call (model, role, latency, chars) to an internal deque
      accessible as brain._llm_request_log.

    The router always returns a client pointing at the registered port for
    the primary model that has `role` in its `default_for` list.  If no
    model is registered as default for the role, falls back to any model
    whose `roles` list contains the role.  If still nothing, uses a plain
    LLMClient on the default host:port from environment.
    """

    _REQUEST_LOG_MAXLEN = 500

    # Hysteresis: an OFFLINE model must out-score the best adequate HEALTHY
    # model by at least this much before we prefer (and request to start) it.
    # Below the margin we keep using the healthy model rather than paying a
    # cold start for a marginal quality gain — the core "grass is greener" fix.
    _OFFLINE_PREFER_MARGIN = float(os.getenv("LLM_OFFLINE_PREFER_MARGIN", "0.15"))
    # Minimum seconds between ensure_role posts for the same (role, model), and
    # the longer back-off applied after a request is declined (RAM / cooldown).
    _ENSURE_ROLE_COOLDOWN_S = float(os.getenv("LLM_ENSURE_ROLE_COOLDOWN_S", "60"))
    _ENSURE_ROLE_REJECT_BACKOFF_S = float(
        os.getenv("LLM_ENSURE_ROLE_REJECT_BACKOFF_S", "180"))

    def __init__(self, registry: List[Dict], brain: "IyyeBrain") -> None:
        self._registry: List[Dict] = registry
        self._brain = brain
        self._request_log: Deque[Dict] = collections.deque(
            maxlen=self._REQUEST_LOG_MAXLEN
        )
        # Per-model counters: model_name -> {requests_total, total_latency_s, last_request}
        self._model_stats: Dict[str, Dict] = {}
        # Set of ports currently confirmed healthy by LlmManagementStream.
        # None means "not yet checked" — _find_model falls back to default_for order.
        self._healthy_ports: Optional[set] = None
        # Per-role model overrides: role -> model_name.  When set, _find_model
        # returns this model instead of following default_for / roles logic.
        self._role_overrides: Dict[str, str] = {}
        # Serialises _record / get_stats_summary: with the async LLM scheduler,
        # worker threads call _record concurrently with the main loop reading
        # stats.  _request_log is a deque (atomic append) but _model_stats is a
        # read-modify-write dict.
        self._stats_lock = threading.Lock()
        # Anti-thrash for ensure_role posts ("grass is greener" fix): a
        # (role, model_name) -> monotonic deadline map.  A request for the same
        # (role, model) is suppressed until its deadline, so N resolutions in a
        # row no longer produce N mailbox messages, and a declined request backs
        # off instead of re-posting every tick.
        self._ensure_role_cooldown: Dict[tuple, float] = {}
        # Expose on brain for other streams
        brain._llm_request_log = self._request_log

    def update_healthy_ports(self, ports: set) -> None:
        """Called by LlmManagementStream after each health check tick."""
        self._healthy_ports = ports

    def mark_port_unhealthy(self, port: Optional[int]) -> None:
        """Drop *port* from the healthy set after a live connection failure.

        Health checks only run every ~30 ticks, so a model that died (e.g.
        OOM-killed under memory pressure) stays in ``_healthy_ports`` until the
        next probe — and the router keeps routing requests to the dead port.
        Scheduler worker threads call this on a connection error so subsequent
        requests reroute immediately; the next health check re-probes and
        restores the port if it recovered.  ``set.discard`` is atomic under the
        GIL, so this is safe to call off the main thread."""
        hp = self._healthy_ports
        if hp is not None and port is not None and port in hp:
            hp.discard(port)
            log.info("LLMRouter: port %s marked unhealthy after a connection "
                     "failure — rerouting until next health check", port)

    def set_role_override(self, role: str, model_name: str) -> bool:
        """Override the model used for *role* until cleared.

        Returns True if the model exists in the registry, False otherwise.
        """
        entry = next((m for m in self._registry if m["name"] == model_name), None)
        if entry is None:
            return False
        self._role_overrides[role] = model_name
        log.info("LLMRouter: role %r overridden → %s (port %d)", role, model_name, entry["port"])
        return True

    def clear_role_override(self, role: str) -> None:
        """Remove a role override, returning to normal default_for routing."""
        removed = self._role_overrides.pop(role, None)
        if removed:
            log.info("LLMRouter: role %r override cleared (was %s)", role, removed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_client(
        self,
        role: str = "chat",
        conscious: bool = False,
        task: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Return a proxy LLMClient (or VisionClient) for the given role.

        When *conscious* is True the most powerful healthy non-vision model is
        returned regardless of role, giving the focused stream exclusive access
        to the best model.  Subconscious callers (conscious=False, the default)
        are routed normally but are excluded from using that top model.

        *task* is an optional spec that enables quantitative model scoring::

            task={
                "prompt_tokens":          1200,
                "expected_output_tokens": 150,
                "quality_need":           0.3,   # 0-1
                "latency_budget_s":       8,
                "urgency":                0.4,   # 0-1
            }

        When *task* is provided the router scores every registry model and
        picks the one with the highest utility/cost ratio.  When omitted
        the original default_for / roles priority logic is used.

        kwargs are forwarded to LLMClient.__init__ (e.g. no_think, max_tokens).
        Each call creates a fresh underlying client — callers should cache the
        returned proxy if they want to reuse it across ticks.

        Equivalent to ``build_client(resolve(...), ...)`` — the two steps are
        exposed separately so the async LLM scheduler can resolve a model (to
        pick a port queue) on the main thread and build the client on a worker
        thread.
        """
        model = self.resolve(role, conscious=conscious, task=task)
        if model is None:
            log.warning("LLMRouter: no registry model for role=%r — using env defaults", role)
            from llm_client import LLMClient
            return LLMClient(**kwargs)
        return self.build_client(model, role, **kwargs)

    def resolve(
        self,
        role: str = "chat",
        conscious: bool = False,
        task: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """Select the registry model that should serve this request.

        Returns the model dict (with ``port``), or None to mean "use env
        defaults".  When the chosen model is offline, posts an ``ensure_role``
        start request and returns a healthy fallback so the caller never hits a
        connection error.  Side-effecting (mailbox post) but does no I/O — safe
        to call on the main loop thread."""
        if conscious:
            model = self._find_top_model()
        else:
            model = self._find_model(role, task=task)

        if model is not None and self._healthy_ports is not None:
            if model["port"] not in self._healthy_ports:
                self._request_ensure_role(role, model, task)
                fallback = self._find_healthy_fallback(role)
                if fallback is not None:
                    log.info(
                        "LLMRouter: best model %s for role=%r offline, "
                        "using healthy fallback %s while it starts",
                        model["name"], role, fallback["name"],
                    )
                    model = fallback
                # else: no healthy fallback — caller will get a connection
                # error but we've already posted a start request for next time
        return model

    def build_client(self, model: Dict, role: str = "chat", **kwargs):
        """Construct a recording proxy client for an already-resolved *model*.

        Split out of get_client so the scheduler can build the client on a
        worker thread.  May perform the (lazy) underlying client import; does
        not itself make a network request."""
        port     = model["port"]
        host     = os.environ.get("LLM_HOST", "127.0.0.1")
        api_base = f"http://{host}:{port}/v1"

        # Scale the read timeout by model size — larger models decode slower
        # and a premature timeout wastes the entire generation.
        params_b = model.get("parameters_b", 0)
        if params_b >= 100:
            timeout = 600   # 10 min for 100B+ models
        elif params_b >= 25:
            timeout = 300   # 5 min for 25-100B models
        else:
            timeout = 120   # 2 min for small models

        if model.get("vision"):
            from llm_vision_client import VisionClient
            client = VisionClient(api_base=api_base)
            return _VisionClientProxy(client, model["name"], role, self)
        else:
            from llm_client import LLMClient
            client = LLMClient(api_base=api_base, timeout=timeout, **kwargs)
            return _LLMClientProxy(client, model["name"], role, self)

    def get_model_info(self, role: str) -> Optional[Dict]:
        """Return the registry entry for the primary model serving a role."""
        return self._find_model(role)

    def get_request_log(self) -> List[Dict]:
        """Return a snapshot of recent LLM requests (newest last)."""
        return list(self._request_log)

    def get_stats_summary(self) -> Dict:
        """Return per-model request stats for reporting."""
        summary = {}
        with self._stats_lock:
            for name, s in self._model_stats.items():
                total = s["requests_total"]
                summary[name] = {
                    "requests_total": total,
                    "avg_latency_s":  round(s["total_latency_s"] / total, 2) if total else 0,
                    "last_request":   s.get("last_request"),
                }
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # MoE models activate only a fraction of parameters per token, so raw
    # parameters_b overstates their effective quality relative to dense models
    # of the same size.  This multiplier is applied wherever parameters_b is
    # used as a quality proxy (top-model selection and scoring).
    _MOE_QUALITY_DISCOUNT = 0.6

    @staticmethod
    def _effective_params_b(model: Dict) -> float:
        """Return quality-adjusted parameter count (dense > MoE at same size)."""
        params = model.get("parameters_b", 0)
        arch = model.get("architecture", "")
        if "moe" in arch.lower():
            return params * LLMRouter._MOE_QUALITY_DISCOUNT
        return float(params)

    # --- Optimistic initialization + dynamic measurement (HLD: each LLM has
    # prefill/decode tokens-per-second performance properties) -------------
    #
    # An unmeasured model is assumed FAST (optimistic prior) so the router
    # tries it at least once; _record() then regresses its true prefill/decode
    # throughput from observed requests and the confidence blend drags the
    # estimate from the prior toward the measurement.  Priors are bounded so
    # optimism breaks speed ties without zeroing the latency term — and the
    # startup penalty in _score_model still keeps a cold unmeasured model off
    # tight-budget interactive turns, so exploration concentrates on warm
    # models and loose-budget roles.
    _OPTIMISTIC_PREFILL_TPS = 500.0
    _OPTIMISTIC_DECODE_TPS = 80.0
    # Per-sample decay on the regression accumulators (recency-weights recent
    # requests; ~1/(1-decay) effective memory).
    _STATS_DECAY = 0.9
    # Decayed sample count at which the measurement is fully trusted.
    _CONFIDENCE_FULL = 4.0
    # A measurement this old (no recent requests) has decayed halfway back to
    # the optimistic prior, so a stale or one-off reading gets re-probed.
    _STALE_HALFLIFE_S = 6 * 3600

    @staticmethod
    def _solve_throughput(sp2, spr, sr2, spl, srl):
        """Least-squares fit (no intercept) of
        ``latency ≈ prompt_tok/prefill_tps + resp_tok/decode_tps`` over the
        accumulated request samples.  Returns ``(prefill_tps, decode_tps)`` or
        None when ill-conditioned (e.g. <2 samples, or all the same
        prompt:response ratio) or non-physical (negative per-token time)."""
        det = sp2 * sr2 - spr * spr
        if abs(det) < 1e-9:
            return None
        a = (sr2 * spl - spr * srl) / det   # s per prompt token  = 1/prefill_tps
        b = (sp2 * srl - spr * spl) / det   # s per response token = 1/decode_tps
        if a <= 1e-9 or b <= 1e-9:
            return None
        return 1.0 / a, 1.0 / b

    @classmethod
    def _confidence(cls, n_eff: float, last_request, now=None) -> float:
        """How much to trust a model's measurement vs the optimistic prior:
        grows with (decayed) sample count, shrinks as the data ages."""
        if n_eff <= 0 or not last_request:
            return 0.0
        w = min(1.0, n_eff / cls._CONFIDENCE_FULL)
        try:
            last = datetime.fromisoformat(last_request)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age = ((now or datetime.now(timezone.utc)) - last).total_seconds()
        except (ValueError, TypeError):
            return w
        recency = 0.5 ** (max(0.0, age) / cls._STALE_HALFLIFE_S)
        return w * recency

    def _blend_tps(self, model: Dict, kind: str) -> float:
        """Effective prefill/decode tps: registry value if set, else the
        measurement blended toward the optimistic prior by confidence."""
        reg = model.get(f"{kind}_tps", 0)
        if reg and reg > 0:
            return float(reg)
        prior = (self._OPTIMISTIC_PREFILL_TPS if kind == "prefill"
                 else self._OPTIMISTIC_DECODE_TPS)
        stats = self._model_stats.get(model.get("name", ""))
        if not stats:
            return prior
        measured = stats.get(f"observed_{kind}_tps", 0.0)
        if not measured or measured <= 0:
            return prior
        w = self._confidence(stats.get("n_eff", 0.0), stats.get("last_request"))
        return w * measured + (1.0 - w) * prior

    def _effective_prefill_tps(self, model: Dict) -> float:
        return self._blend_tps(model, "prefill")

    def _effective_decode_tps(self, model: Dict) -> float:
        return self._blend_tps(model, "decode")

    def _best_by_throughput(self, candidates: List[Dict]) -> Optional[Dict]:
        """Pick the candidate with highest effective decode throughput."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda m: self._effective_decode_tps(m))

    # ------------------------------------------------------------------
    # Quantitative scoring  (used when task spec is provided)
    # ------------------------------------------------------------------

    # Default task spec used when callers omit it — maps to the legacy
    # "first healthy model" behaviour via very permissive parameters.
    _DEFAULT_TASK: Dict[str, Any] = {
        "prompt_tokens": 500,
        "expected_output_tokens": 200,
        "quality_need": 0.5,
        "latency_budget_s": 60,
        "urgency": 0.5,
    }

    # Importance weights for each role — used as a multiplier in scoring.
    _ROLE_IMPORTANCE: Dict[str, float] = {
        "chat": 1.0, "reasoning": 0.9, "code": 0.8, "codegen": 0.7,
        "vision": 0.7, "stm": 0.3, "alignment": 0.3, "fast": 0.2,
    }

    def _score_model(
        self, model: Dict, role: str, task: Dict[str, Any],
    ) -> float:
        """Compute a quantitative utility score for *model* serving *role*.

        Higher is better.  The formula balances quality, speed, and cost::

            task_value   = urgency × quality_need × role_importance
            model_score  = task_value × model_quality_for_role
                         - latency_penalty
                         - resource_cost
        """
        prompt_tok   = task.get("prompt_tokens", 500)
        output_tok   = task.get("expected_output_tokens", 200)
        quality_need = task.get("quality_need", 0.5)
        budget_s     = task.get("latency_budget_s", 60)
        urgency      = task.get("urgency", 0.5)

        # --- Quality ---
        quality_map = model.get("quality", {})
        model_quality = quality_map.get(role, 0.1)  # low default for unrated roles
        # MoE discount: at equal hand-tuned quality, prefer dense models.
        arch = model.get("architecture", "")
        if "moe" in arch.lower():
            model_quality *= self._MOE_QUALITY_DISCOUNT
        role_importance = self._ROLE_IMPORTANCE.get(role, 0.5)
        task_value = urgency * quality_need * role_importance
        quality_score = task_value * model_quality  # 0..1 range

        # --- Latency ---
        # Effective tps embeds optimistic prior + dynamic measurement, so an
        # unmeasured model looks fast (gets explored) and a measured one is
        # ranked on its real throughput.  Both are always > 0, so the old
        # conservative fallbacks are no longer reached.
        prefill_tps = self._effective_prefill_tps(model) or 50.0
        decode_tps  = self._effective_decode_tps(model) or 15.0
        prefill_s = prompt_tok / prefill_tps
        decode_s = output_tok / decode_tps

        hp = self._healthy_ports
        is_healthy = hp is not None and model["port"] in hp
        startup_s = 0.0 if is_healthy else model.get("startup_s", 45)
        estimated_latency = prefill_s + decode_s + startup_s

        # Penalty: how much the estimated latency exceeds the budget
        if budget_s > 0 and estimated_latency > budget_s:
            latency_penalty = 0.3 * (estimated_latency - budget_s) / budget_s
        else:
            latency_penalty = 0.0

        # --- Resource cost ---
        ram_gb = model.get("ram_required_gb", model.get("size_gb", 20))
        # Normalise to a 0-1 range assuming 128GB total (adjustable)
        resource_cost = 0.1 * (ram_gb / 128.0)
        # Extra penalty for starting an offline model
        if not is_healthy:
            resource_cost += 0.15

        return quality_score - latency_penalty - resource_cost

    def _find_healthy_fallback(self, role: str) -> Optional[Dict]:
        """Return any healthy model that can serve *role*, ignoring scoring."""
        hp = self._healthy_ports
        if hp is None:
            return None
        top = self._find_top_model()
        top_port = top["port"] if top else None

        # Prefer default_for, then roles, then any healthy
        for attr in ("default_for", "roles"):
            cands = [m for m in self._registry
                     if role in m.get(attr, [])
                     and m["port"] in hp
                     and m["port"] != top_port]
            if cands:
                return self._best_by_throughput(cands)
        # Any healthy non-top
        cands = [m for m in self._registry
                 if m["port"] in hp and m["port"] != top_port]
        if cands:
            return self._best_by_throughput(cands)
        # Last resort: top model
        if top and top["port"] in hp:
            return top
        return None

    def _request_ensure_role(
        self, role: str, model: Dict, task: Optional[Dict] = None,
    ) -> None:
        """Post an ``ensure_role`` message to LlmManagementStream, deduplicated.

        At most one post per ``(role, model)`` per ``_ENSURE_ROLE_COOLDOWN_S``,
        so a burst of resolutions (e.g. 5 fast + 5 chat) no longer floods the
        mailbox with 10 start requests."""
        brain = self._brain
        if brain is None:
            return
        key = (role, model["name"])
        now = time.monotonic()
        if now < self._ensure_role_cooldown.get(key, 0.0):
            log.debug("LLMRouter: ensure_role for %s (role=%r) suppressed by cooldown",
                      model["name"], role)
            return
        self._ensure_role_cooldown[key] = now + self._ENSURE_ROLE_COOLDOWN_S
        from messaging import Messages
        brain.post_message("llm_management", Messages.ensure_role(
            role=role,
            model_name=model["name"],
            task=task,
            reason=f"get_client(role={role!r}) — best model {model['name']} not healthy",
        ))
        log.info(
            "LLMRouter: requested ensure_role for %s (role=%r)",
            model["name"], role,
        )

    def note_ensure_role_rejected(self, role: str, model_name: str) -> None:
        """Called by LlmManagementStream when an ensure_role request is declined
        (RAM pressure, eviction cooldown).  Applies a longer back-off so the
        router doesn't keep asking for a model the manager just refused."""
        self._ensure_role_cooldown[(role, model_name)] = (
            time.monotonic() + self._ENSURE_ROLE_REJECT_BACKOFF_S
        )

    def _find_top_model(self) -> Optional[Dict]:
        """Return the most powerful healthy non-vision model.

        Primary key: effective parameters (dense > MoE at same size).
        Tiebreaker: decode throughput.
        """
        hp = self._healthy_ports
        candidates = [
            m for m in self._registry
            if not m.get("vision") and (hp is None or m["port"] in hp)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: (
            self._effective_params_b(m), self._effective_decode_tps(m),
        ))

    def _find_model(
        self, role: str, task: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """Select the best model for *role*, optionally using a task spec.

        When *task* is provided, every candidate model is scored
        quantitatively and the highest-scoring one is returned — even if
        it is currently offline (the caller handles startup).

        When *task* is ``None`` the legacy priority logic is used:
        default_for → roles → any healthy → top model.
        """
        # Check for a user-set override first (e.g. "use qwen36-27b for chat").
        override_name = self._role_overrides.get(role)
        if override_name:
            entry = next((m for m in self._registry if m["name"] == override_name), None)
            if entry is not None:
                hp = self._healthy_ports
                if hp is None or entry["port"] in hp:
                    return entry
                log.warning(
                    "LLMRouter: override model %s for role=%r is not healthy, "
                    "falling through to normal routing",
                    override_name, role,
                )

        hp = self._healthy_ports  # None = not yet checked
        # Reserve the top model for the conscious stream — BUT only when there
        # is at least one *other* healthy non-vision model to serve subconscious
        # work.  If the top model is the only healthy one, excluding it here
        # forces every subconscious resolution to pick an offline model and post
        # an ensure_role (the "grass is greener" storm); in that case allow it.
        top = self._find_top_model()
        top_port = top["port"] if top else None
        other_healthy = [
            m for m in self._registry
            if not m.get("vision")
            and hp is not None and m["port"] in hp
            and m["port"] != top_port
        ]
        reserve_top = top_port is not None and len(other_healthy) >= 1

        def _ok(m: Dict) -> bool:
            return (not reserve_top) or m["port"] != top_port

        # ---- Quantitative scoring path ----
        # When health is known, always score — even if the caller didn't
        # supply an explicit task spec.  This ensures every get_client()
        # call benefits from quality/latency/resource scoring instead of
        # just picking the first healthy default_for match.
        effective_task = task
        if effective_task is None and hp is not None:
            effective_task = dict(self._DEFAULT_TASK)
            # Tune defaults per role so the score differentiates properly.
            importance = self._ROLE_IMPORTANCE.get(role, 0.5)
            effective_task["quality_need"] = max(0.3, importance)
            effective_task["urgency"] = importance

        if effective_task is not None:
            # Candidate pool: models that list the role PLUS any healthy model.
            # Including healthy models (even ones that don't list the role) lets
            # the scorer keep using an already-loaded model instead of always
            # requesting an offline role-specialist — the "grass is greener"
            # fix.  _ok() still applies the top-model reservation.
            candidates = [
                m for m in self._registry
                if (role in m.get("roles", [])
                    or (hp is not None and m["port"] in hp))
                and _ok(m)
            ]
            if not candidates:
                candidates = [m for m in self._registry if _ok(m)]
            if not candidates:
                return top
            scored = [(self._score_model(m, role, effective_task), m)
                      for m in candidates]
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best = scored[0]
            if log.isEnabledFor(logging.DEBUG):
                top3 = [(round(s, 4), m["name"]) for s, m in scored[:3]]
                log.debug(
                    "LLMRouter: scored %d models for role=%r → %s",
                    len(scored), role, top3,
                )
            # Hysteresis: if the top-scoring candidate is OFFLINE, only prefer
            # it over the best adequate HEALTHY candidate when it wins by a
            # substantial margin.  Otherwise keep using the healthy model rather
            # than paying a cold start (and posting an ensure_role) for a
            # marginal gain — the "grass is greener" fix.
            hp_set = hp or set()
            if best["port"] not in hp_set:
                healthy = [(s, m) for s, m in scored if m["port"] in hp_set]
                if healthy:
                    h_score, h_model = healthy[0]
                    if best_score - h_score < self._OFFLINE_PREFER_MARGIN:
                        log.debug(
                            "LLMRouter: keeping healthy %s (%.3f) over offline "
                            "%s (%.3f) — within %.2f margin",
                            h_model["name"], h_score, best["name"], best_score,
                            self._OFFLINE_PREFER_MARGIN,
                        )
                        return h_model
            return best

        # ---- Legacy priority path (health not yet known) ----
        # This only runs before the first health check completes.
        # Once _healthy_ports is set, the scoring path above handles everything.
        cands = [m for m in self._registry
                 if role in m.get("default_for", []) and _ok(m)]
        if cands:
            return self._best_by_throughput(cands)
        cands = [m for m in self._registry
                 if role in m.get("roles", []) and _ok(m)]
        if cands:
            return self._best_by_throughput(cands)
        # Absolute fallback: top model is better than None
        return top

    def _record(
        self,
        model_name: str,
        role: str,
        prompt_chars: int,
        response_chars: int,
        latency_s: float,
    ) -> None:
        entry = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "model":          model_name,
            "role":           role,
            "prompt_chars":   prompt_chars,
            "response_chars": response_chars,
            "latency_s":      round(latency_s, 2),
        }
        # Worker threads (LLM scheduler) call _record concurrently with the
        # main loop; serialise the deque append + stats read-modify-write.
        p_tok = prompt_chars / 4.0     # ~4 chars/token
        r_tok = response_chars / 4.0
        with self._stats_lock:
            self._request_log.append(entry)

            s = self._model_stats.setdefault(model_name, {
                "requests_total": 0,
                "total_latency_s": 0.0,
                "last_request": None,
                "observed_prefill_tps": 0.0,
                "observed_decode_tps": 0.0,
                # Decayed sufficient statistics for the throughput regression
                # (latency ≈ p_tok/prefill_tps + r_tok/decode_tps).
                "n_eff": 0.0, "sp2": 0.0, "spr": 0.0,
                "sr2": 0.0, "spl": 0.0, "srl": 0.0,
            })
            s["requests_total"]  += 1
            s["total_latency_s"] += latency_s
            s["last_request"]     = entry["timestamp"]

            # Fold this request into the regression — recency-weighted by
            # decaying the running sums first.  Separating prefill from decode
            # statistically fixes the old "(resp_chars/4)/total_latency" bias,
            # which folded prefill + queue wait into "decode tps".
            if r_tok > 5 and latency_s > 0.05:
                d = self._STATS_DECAY
                for k in ("n_eff", "sp2", "spr", "sr2", "spl", "srl"):
                    s[k] = s.get(k, 0.0) * d
                s["n_eff"] += 1.0
                s["sp2"]   += p_tok * p_tok
                s["spr"]   += p_tok * r_tok
                s["sr2"]   += r_tok * r_tok
                s["spl"]   += p_tok * latency_s
                s["srl"]   += r_tok * latency_s
                sol = self._solve_throughput(
                    s["sp2"], s["spr"], s["sr2"], s["spl"], s["srl"])
                if sol is not None:
                    pf, dc = sol
                    s["observed_prefill_tps"] = min(5000.0, pf)
                    s["observed_decode_tps"]  = min(1000.0, dc)

        log.debug(
            "LLMRouter: model=%s role=%s latency=%.1fs prompt=%d resp=%d",
            model_name, role, latency_s, prompt_chars, response_chars,
        )


# ======================================================================
# LlmManagementStream
# ======================================================================

class LlmManagementStream(ProcessingStream):
    """
    Monitors all registered LLMs, optionally starts them when unhealthy,
    writes tools/llm-active.json every health tick, and hosts LLMRouter.
    """

    HEALTH_INTERVAL       = 30   # ticks between health checks
    _AUTO_START_THRESHOLD = 2    # consecutive failures before auto-start

    def __init__(self, brain: "IyyeBrain") -> None:
        super().__init__(name="llm_management")
        self.brain = brain
        self.priority = 1
        self._can_be_conscious = False

        # Load registry
        self._registry: List[Dict] = self._load_registry()

        # Create router and expose on brain
        self._router = LLMRouter(self._registry, brain)
        brain.llm_router = self._router

        # Async LLM job scheduler — moves blocking LLM calls off the main loop
        # (HLD issue #3).  Exposed on the brain so any stream can submit jobs.
        # Per-port limit 1: llama.cpp is effectively single-slot and parallel
        # slots share the KV budget that drives memory pressure.
        from llm_scheduler import LLMScheduler
        self._scheduler = LLMScheduler(self._router, brain, per_port_limit=1)
        brain.llm_scheduler = self._scheduler
        # Adopt the brain's current wake epoch (this stream is created during
        # WAKING_UP, after _enter_waking_up has already bumped the epoch).
        self._scheduler.on_wake(getattr(brain, "_wake_epoch", 0))

        # Per-model failure counters and start-in-progress flags
        self._failures:    Dict[str, int]  = {}
        self._starting:    Dict[str, bool] = {}
        # Models a stop has been issued for but not yet confirmed down by a
        # health check.  The per-tick memory guard treats these as already
        # freed so it doesn't re-issue stops (thrash) against a stale status.
        self._stopped:     set = set()
        # Anti-thrash timestamps (monotonic):
        #   _started_at   — when a start was last initiated; protects a model
        #                   from eviction for _MIN_RESIDENCY_S (minimum residency).
        #   _evicted_at   — when a model was last stopped for memory pressure;
        #                   blocks restarting it for _EVICTION_COOLDOWN_S so an
        #                   evict→request→evict loop can't form.
        self._started_at:  Dict[str, float] = {}
        self._evicted_at:  Dict[str, float] = {}

        self._ticks_since_check: int = self.HEALTH_INTERVAL  # check on first tick
        self._last_all_status:   List[Dict] = []
        # Health checks (N sequential /health probes + RSS reads) run on a
        # background thread so they never stall the main loop (issue #8); the
        # raw result is applied on a later tick by _apply_health_result.
        self._health_thread:  Optional[threading.Thread] = None
        self._health_result:  Optional[List[Dict]] = None

        self._host        = os.getenv("LLM_HOST", "127.0.0.1")
        self._auto_start  = bool(os.getenv("LLM_AUTO_START", ""))
        self._vision_auto = bool(os.getenv("LLM_VISION_AUTO_START", ""))

        # Minimum available RAM (GB) before shedding a low-value model.
        self._HEADROOM_MIN_GB = 4.0
        # Critical floor: below this we are about to OOM, so the normal
        # protections (never evict the top/conscious model; never stop the last
        # model) are lifted as a last resort — degraded chat beats an OOM-kill
        # of the whole process.  Must be below _HEADROOM_MIN_GB.
        self._HEADROOM_CRITICAL_GB = float(os.getenv("LLM_HEADROOM_CRITICAL_GB", "2.0"))
        # Minimum ratio of available RAM to loaded model memory.
        # When available_gb / loaded_gb drops below this, the system is
        # over-committed and should shed a model.  A ratio of 0.6 means
        # "start worrying when free RAM is less than 60% of what models
        # are consuming."  This adapts to varying baseline system usage
        # unlike a fixed percentage threshold.
        self._HEADROOM_RATIO = 0.6
        # Loading a model transiently uses more RAM than its steady-state size
        # (weight mmap + context/KV-cache init).  The start gate reserves this
        # spike on top of the model size so a load can't OOM before the
        # per-tick shedder can react.  Margin = max(MIN_GB, size * FRACTION).
        self._LOAD_SPIKE_FRACTION = float(os.getenv("LLM_LOAD_SPIKE_FRACTION", "0.2"))
        self._LOAD_SPIKE_MIN_GB   = float(os.getenv("LLM_LOAD_SPIKE_MIN_GB", "2.0"))
        # Anti-thrash windows (seconds).  A model just memory-evicted may not be
        # restarted within _EVICTION_COOLDOWN_S; a model just started may not be
        # evicted within _MIN_RESIDENCY_S (unless under critical OOM pressure).
        self._EVICTION_COOLDOWN_S = float(os.getenv("LLM_EVICTION_COOLDOWN_S", "120"))
        self._MIN_RESIDENCY_S     = float(os.getenv("LLM_MIN_RESIDENCY_S", "90"))
        # Restart debounce margin: a model started within startup_s + this many
        # seconds is NOT restarted again, covering the window between the start
        # script returning and the periodic health check confirming the model
        # healthy.  Without it, repeated "unreachable" detections in that window
        # each fire a redundant restart (the observed flapping).
        self._RESTART_GRACE_S = float(os.getenv("LLM_RESTART_GRACE_S", "30"))

        # ---- KV-cache-aware memory accounting (issue #5) ----
        # registry size_gb is weights-only.  llama-server is launched with a
        # large context (LLM_CONTEXT, default 64k) split across LLM_PARALLEL
        # slots, so the KV cache adds GBs that weights-only sizing never sees —
        # which is how models loaded past physical RAM.  We measure each running
        # model's real RSS (weights + KV + overhead) and use that everywhere;
        # the observed peak also self-calibrates the start estimate after a
        # model has run once.  For a never-seen model the start estimate adds a
        # coarse KV term computed from the actual launch context budget.
        self._observed_rss_gb: Dict[str, float] = {}
        self._launch_context_k = float(os.getenv("LLM_CONTEXT", "65536")) / 1024.0
        self._launch_parallel  = float(os.getenv("LLM_PARALLEL", "4"))
        # GB of KV cache per 1K tokens of total context, at the reference model
        # size; scaled linearly by parameters_b.  Coarse first-run estimate only
        # (superseded by measured RSS); tune via env if your models differ.
        self._KV_GB_PER_K_CTX     = float(os.getenv("LLM_KV_GB_PER_K_CTX", "0.06"))
        self._KV_REFERENCE_PARAMS = float(os.getenv("LLM_KV_REFERENCE_PARAMS_B", "30"))

        # Override default model script via env (legacy compat)
        env_default = os.getenv("LLM_DEFAULT_MODEL", "")
        if env_default:
            for m in self._registry:
                if "chat" in m.get("default_for", []):
                    m["script"] = env_default
                    break

    def can_stop_safely(self) -> bool:
        return not any(self._starting.values())

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Skip while paused (winding-down): no health checks, no auto-start.
        # Existing LLM servers keep running for sleep replay; we only freeze
        # management commands at the phase boundary.
        if self._paused:
            # Winding-down: no health checks or auto-start (those spawn
            # background work pause forbids).  Still deliver any urgent,
            # pause-safe control messages; non-urgent ones stay queued for
            # the next awake tick instead of being silently consumed.
            for msg in self.brain.drain_messages("llm_management", urgent_only=True):
                self._handle_message(msg)
            return None

        # Process inter-stream messages (e.g. restart requests from self-reflection).
        for msg in self.brain.drain_messages("llm_management"):
            self._handle_message(msg)

        # Memory-pressure guard runs EVERY tick (cheap: one psutil read + cached
        # model sizes, no health probes).  Model starts are processed per tick,
        # so shedding must also be per tick — gating it behind the 30-tick
        # health check let RAM climb into OOM territory between checks.
        self._relieve_memory_pressure(self._last_all_status or [])

        # Apply a completed background health check (probing ran off the main
        # loop — issue #8).
        applied = self._apply_health_result()

        # Periodically kick off the next background health check.
        self._ticks_since_check += 1
        if (self._ticks_since_check >= self.HEALTH_INTERVAL
                and self._health_thread is None):
            self._ticks_since_check = 0
            self._start_health_check()

        self.checkpoint()
        return {"llm_status": self._last_all_status} if applied else None

    # ------------------------------------------------------------------
    # Background health checks (issue #8 — never block the main loop)
    # ------------------------------------------------------------------

    def _start_health_check(self) -> None:
        """Spawn the N-model health probe on a background thread."""
        if self._paused:
            return
        self._health_result = None

        def _run() -> None:
            try:
                result = self._check_all_models()
            except Exception as exc:
                log.warning("LlmManagementStream: health check failed: %s", exc)
                result = None
            self._health_result = result

        t = threading.Thread(target=_run, name="llm_health_check", daemon=True)
        # Prune finished background threads (health/start/stop) so the list
        # doesn't grow across a long awake cycle.
        self._background_threads = [x for x in self._background_threads if x.is_alive()]
        self._background_threads.append(t)
        self._health_thread = t
        t.start()

    def _apply_health_result(self) -> bool:
        """If the background health check has finished, apply its result on the
        main thread.  Returns True when a result was applied."""
        t = self._health_thread
        if t is None or t.is_alive():
            return False
        self._health_thread = None
        all_status = self._health_result
        self._health_result = None
        if not all_status:
            return False

        self._last_all_status = all_status
        # Fresh probe is now authoritative — drop the pending-stop shadow set.
        self._stopped.clear()

        # Backward-compat: expose first chat model status on brain._llm_status
        for s in all_status:
            if "chat" in s.get("default_for", []):
                self.brain._llm_status = s
                break

        # Update router's healthy-port set so get_client() routes correctly
        self._router.update_healthy_ports(
            {s["port"] for s in all_status if s["healthy"]}
        )

        # Write llm-active.json for hardware sensor and self-reflection
        self._write_active_json(all_status)

        # Log status changes and trigger auto-start where needed
        for s in all_status:
            name = s["name"]
            healthy = s["healthy"]
            if healthy:
                if self._failures.get(name, 0) > 0:
                    self.add_to_log(
                        f"LLM {name} back online (port {s['port']})"
                    )
                elif name not in self._failures:
                    self.add_to_log(
                        f"LLM healthy: {name} port={s['port']} "
                        f"size={s['size_gb']}GB"
                    )
                self._failures[name] = 0
            else:
                self._failures[name] = self._failures.get(name, 0) + 1
                # Only log the first failure and threshold crossings
                if self._failures[name] == 1:
                    self.add_to_log(
                        f"LLM {name} not healthy (port {s['port']})"
                    )
                should_auto = (
                    self._auto_start
                    and not self._starting.get(name, False)
                    # Don't re-fire a start in the post-start, pre-healthy
                    # window — the same debounce that stops the restart-message
                    # flapping (the health check just hasn't confirmed yet).
                    and not self._recently_started(name)
                    and self._failures[name] >= self._AUTO_START_THRESHOLD
                    and ("chat" in s.get("default_for", [])
                         or (self._vision_auto and s.get("vision")))
                )
                if should_auto:
                    self._try_start(s)

        # Proactive resource management against the freshly-probed status.
        self._relieve_memory_pressure(all_status)
        return True

    # ------------------------------------------------------------------
    # Inter-stream message handling
    # ------------------------------------------------------------------

    def _recently_started(self, name: str) -> bool:
        """True if a start for *name* began within its startup + grace window —
        long enough that the periodic health check has not yet confirmed it.
        Reuses ``_started_at`` (set at start initiation)."""
        last = self._started_at.get(name)
        if not last:
            return False
        entry = next((m for m in self._registry if m["name"] == name), None)
        cooldown = (entry.get("startup_s", 45) if entry else 45) + self._RESTART_GRACE_S
        return (time.monotonic() - last) < cooldown

    def chat_llm_state(self) -> str:
        """Authoritative state of the default chat model, for observers.

        The health owner (this stream) derives state from the scheduler's
        first-hand ``model_unavailable`` evictions (``_healthy_ports``) plus its
        own restart bookkeeping — so self-reflection consults THIS instead of
        acting on a bare "unreachable" snapshot that races an in-flight start:

        - ``healthy``   — the chat model's port is in the healthy set.
        - ``restoring`` — a start is in flight or within the post-start grace
                          window (the owner is already handling it).
        - ``down``      — unreachable and no restart underway (escalate).
        """
        model = self._router._find_model("chat")
        if model is None:
            return "down"
        name = model["name"]
        hp = self._router._healthy_ports
        if hp is not None and model["port"] in hp:
            return "healthy"
        if self._starting.get(name, False) or self._recently_started(name):
            return "restoring"
        return "down"

    def _should_skip_start(self, entry: Dict) -> Optional[str]:
        """Reason to NOT (re)start *entry* right now, or None if a start is
        warranted.  The single debounce gate for every restart trigger
        (message handler, ensure_role, health-loop auto-start): a model that is
        already starting, already healthy, or started within the cooldown must
        not be started again — which is what stops the restart flapping."""
        name = entry["name"]
        if self._starting.get(name, False):
            return "already starting"
        hp = self._router._healthy_ports
        if hp is not None and entry["port"] in hp:
            return "already healthy"
        if self._recently_started(name):
            return "started recently (health check not yet confirmed)"
        return None

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "restart":
            role = msg.get("role", "chat")
            model = self._router._find_model(role)
            if model:
                skip = self._should_skip_start(model)
                if skip:
                    log.debug("restart for role=%r skipped: %s", role, skip)
                else:
                    self.add_to_log(
                        f"Restart requested for role={role!r} "
                        f"({msg.get('reason','?')})"
                    )
                    self._try_start(model)
        elif action == "ensure_role":
            self._handle_ensure_role(msg)
        elif action == "stop":
            name = msg.get("name", "")
            self.stop_model(name)
        elif action == "start":
            script = msg.get("script", "")
            self.start_model(script)
        else:
            log.debug("LlmManagementStream: unknown message action %r", action)

    def _handle_ensure_role(self, msg: Dict[str, Any]) -> None:
        """Decide whether to start a model requested by the router.

        The decision is based on whether the projected benefit (serving the
        role with a dedicated model over many upcoming ticks) outweighs the
        cold-start cost and RAM pressure.
        """
        model_name = msg.get("model_name", "")
        role = msg.get("role", "?")
        reason = msg.get("reason", "")
        entry = next((m for m in self._registry if m["name"] == model_name), None)
        if entry is None:
            log.warning("ensure_role: unknown model %r", model_name)
            return
        # Shared debounce: already starting / healthy / started-recently.
        skip = self._should_skip_start(entry)
        if skip:
            log.debug("ensure_role: %s skipped: %s", model_name, skip)
            return

        # Anti-thrash: refuse to restart a model that was just memory-evicted.
        # Otherwise the router (which preferred it) requests it, memory pressure
        # evicts it, and the next request asks for it again — an expensive
        # evict→start→evict loop.  Back the router off so it stops asking.
        now = time.monotonic()
        evicted_ago = now - self._evicted_at.get(model_name, 0.0)
        if evicted_ago < self._EVICTION_COOLDOWN_S:
            log.info("ensure_role: declining %s — evicted %.0fs ago "
                     "(cooldown %.0fs)", model_name, evicted_ago,
                     self._EVICTION_COOLDOWN_S)
            self.add_to_log(
                f"ensure_role: declined {model_name} for {role} — "
                f"memory-evicted {evicted_ago:.0f}s ago"
            )
            self._router.note_ensure_role_rejected(role, model_name)
            return

        # Check RAM headroom.  ram_needed is the KV-aware full footprint (not
        # weights-only): the observed RSS from a prior run, else size + KV.
        ram_needed = self._start_footprint_gb(entry)
        try:
            import psutil
            ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            ram_avail = 32.0  # optimistic default

        # Budget for the transient RAM spike while the model loads (weight mmap
        # + context/KV init), not just its steady-state size — otherwise a load
        # can OOM before the per-tick shedder reacts.
        spike_margin = max(self._LOAD_SPIKE_MIN_GB,
                           ram_needed * self._LOAD_SPIKE_FRACTION)
        effective_need = ram_needed + spike_margin

        # Loaded RAM = real per-process RSS of healthy models (KV included),
        # not weights-only size_gb.
        all_status = self._last_all_status or []
        loaded_gb = sum(self._running_footprint_gb(s)
                        for s in all_status if s.get("healthy"))
        ratio = ram_avail / loaded_gb if loaded_gb > 0 else 999.0

        # Decline (after trying to evict for the *peak* need) when starting would
        # leave too little headroom, for one of two reasons:
        #   - under_ratio: the system is already over-committed, or
        #   - below_floor: available at the load peak would drop under the
        #     absolute floor.
        # The floor applies to EVERY model — previously a "small" (<10 GB) model
        # bypassed it, which let a 7.6 GB model load alongside a large one and
        # contributed to an OOM-kill.
        under_ratio = loaded_gb > 0 and ratio < self._HEADROOM_RATIO
        below_floor = (ram_avail - effective_need) < self._HEADROOM_MIN_GB
        if under_ratio or below_floor:
            freed = self._try_evict_for(effective_need, ram_avail)
            if freed < effective_need * 0.5:
                why = "memory pressure" if under_ratio else "load-spike headroom"
                log.info(
                    "ensure_role: skipping %s — %s (avail %.1f GB, need %.1f GB "
                    "+%.1f GB spike, loaded %.1f GB, ratio %.2f, freed %.1f GB)",
                    model_name, why, ram_avail, ram_needed, spike_margin,
                    loaded_gb, ratio, freed,
                )
                self.add_to_log(
                    f"ensure_role: declined {model_name} for {role} — {why} "
                    f"(avail {ram_avail:.1f}GB, need {ram_needed:.1f}+{spike_margin:.1f} spike)"
                )
                # Back the router off so it stops re-requesting a model we
                # can't fit right now.
                self._router.note_ensure_role_rejected(role, model_name)
                return

        self.add_to_log(
            f"ensure_role: starting {model_name} for {role} ({reason})"
        )
        self._try_start(entry)

    # ------------------------------------------------------------------
    # Proactive resource management
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # KV-cache-aware memory footprint (issue #5)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_rss_gb(pid: int) -> float:
        """Resident-set size of *pid* in GB (weights + KV + runtime), or 0.0."""
        if not pid:
            return 0.0
        try:
            import psutil
            return psutil.Process(pid).memory_info().rss / 1024 ** 3
        except Exception:
            return 0.0

    @staticmethod
    def _running_footprint_gb(status: Dict) -> float:
        """Real RAM a *running* model holds: measured RSS when available, else
        the weights-only registry size as a (low) fallback."""
        rss = status.get("rss_gb", 0.0) or 0.0
        return rss if rss > 0 else float(status.get("size_gb", 0) or 0)

    def _kv_estimate_gb(self, entry: Dict) -> float:
        """Coarse KV-cache estimate for a model not yet running, from the actual
        launch context budget (LLM_CONTEXT × LLM_PARALLEL) scaled by model size.
        Used only until the model's real RSS has been observed."""
        total_ctx_k = self._launch_context_k * self._launch_parallel
        params = float(entry.get("parameters_b", 0) or 0)
        size_scale = (params / self._KV_REFERENCE_PARAMS) if params > 0 else 1.0
        return total_ctx_k * self._KV_GB_PER_K_CTX * max(0.2, size_scale)

    def _start_footprint_gb(self, entry: Dict) -> float:
        """Estimated full RAM footprint of starting *entry* (weights + KV).

        Prefers the observed peak RSS from a prior run of this model (accurate,
        self-calibrating); otherwise a KV-aware estimate that is at least the
        registry's ``ram_required_gb`` and never below ``size_gb + KV``."""
        observed = self._observed_rss_gb.get(entry["name"], 0.0)
        if observed > 0:
            return observed
        size = float(entry.get("ram_required_gb", entry.get("size_gb", 20)) or 20)
        weights = float(entry.get("size_gb", size) or size)
        return max(size, weights + self._kv_estimate_gb(entry))

    def _eviction_score(self, status: Dict) -> float:
        """Score a running model for eviction — **lower** = more expendable.

        ::

            score = default_role_count * 5
                  + recent_requests_weight   (requests in last 10 min)
                  + conscious_reserved_bonus  (10 if this is the top model)
                  - size_gb_freed_bonus       (larger model → more incentive to free)
        """
        defaults = len(status.get("default_for", []))
        # Recent request weight: use stats from the router
        stats = self._router._model_stats.get(status["name"], {})
        total_req = stats.get("requests_total", 0)
        last_req = stats.get("last_request")
        recency_bonus = 0.0
        if last_req:
            try:
                last_dt = datetime.fromisoformat(last_req)
                age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                if age_min < 10:
                    recency_bonus = (10 - age_min) * 0.5  # up to 5 pts
            except Exception:
                pass
        # Favor freeing the model with the largest *real* footprint (RSS incl.
        # KV cache), not weights-only size — so a small-weights model with a
        # huge KV cache is correctly seen as the RAM hog.
        footprint_gb = self._running_footprint_gb(status)
        return (defaults * 5.0
                + min(total_req, 50) * 0.1  # cap contribution at 5 pts
                + recency_bonus
                - footprint_gb * 0.05)  # bonus for freeing larger footprints

    def _relieve_memory_pressure(self, status: List[Dict]) -> None:
        """Stop the least valuable model when memory pressure is high.

        Triggers eviction when either:
        - available RAM drops below ``_HEADROOM_MIN_GB`` (absolute floor), or
        - available RAM / loaded model memory drops below ``_HEADROOM_RATIO``.

        ``status`` may be a freshly-probed model list (health-check path) or
        the cached ``_last_all_status`` (per-tick guard).  Models already being
        stopped (``_starting`` flag) or whose stop has been issued but not yet
        confirmed down (``_stopped``) are treated as *not* holding RAM, so the
        per-tick guard projects the freeing-in-progress and won't re-shed the
        same model every tick — but it WILL shed an additional model next tick
        if the projected free still isn't enough.

        Normally never stops the top model (reserved for the conscious stream)
        or the last running model.  But under *critical* pressure (available
        below ``_HEADROOM_CRITICAL_GB`` — about to OOM) those protections are
        lifted as a last resort: losing the LLM is degraded but survivable; an
        OOM-kill of the whole process is not.
        """
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / 1024 ** 3
        except Exception:
            return

        # Models actually holding RAM *now*: healthy and not already leaving.
        holding = [
            s for s in status
            if s.get("healthy")
            and not self._starting.get(s["name"], False)
            and s["name"] not in self._stopped
        ]
        # Real loaded RAM = sum of measured per-process RSS (KV included).
        loaded_gb = sum(self._running_footprint_gb(s) for s in holding)
        if loaded_gb <= 0:
            return

        ratio_ok = available_gb / loaded_gb >= self._HEADROOM_RATIO
        if available_gb >= self._HEADROOM_MIN_GB and ratio_ok:
            return

        # Last-resort mode when we're about to OOM.
        critical = available_gb < self._HEADROOM_CRITICAL_GB
        if len(holding) <= 1 and not critical:
            return  # never stop the last running model (unless critical)

        top = self._router._find_top_model()
        top_port = top["port"] if top else None
        now = time.monotonic()
        # Under critical pressure the top/conscious model is also a candidate
        # (it is often the RAM hog — e.g. a large KV cache — that the normal
        # rule would protect to death).  Eviction scoring still prefers the
        # most expendable model, so the top is only chosen if it is the lowest
        # score or the sole remaining model.  Minimum residency: a model
        # started within _MIN_RESIDENCY_S is not evicted (unless critical), so
        # a just-loaded model isn't torn down before it does any work.
        candidates = [
            (self._eviction_score(s), s) for s in holding
            if (critical or s["port"] != top_port)
            and (critical or
                 now - self._started_at.get(s["name"], 0.0) >= self._MIN_RESIDENCY_S)
        ]
        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        victim = candidates[0][1]
        is_top = victim["port"] == top_port
        prefix = "CRITICAL last-resort eviction" if critical else "Memory pressure"
        self.add_to_log(
            f"{prefix} (avail {available_gb:.1f}GB, loaded {loaded_gb:.1f}GB, "
            f"ratio {available_gb / loaded_gb:.2f}) — stopping {victim['name']} "
            f"({victim['size_gb']}GB)"
            + (" [top/conscious model]" if is_top else "")
        )
        # Record the eviction so ensure_role won't immediately re-request it.
        self._evicted_at[victim["name"]] = now
        self._try_stop(victim)

    def _try_evict_for(self, need_gb: float, avail_gb: float) -> float:
        """Try to free *need_gb* of RAM by evicting the least valuable model.

        Returns the amount of RAM (GB) that the evicted model will free
        (actual RAM won't be available immediately — model shutdown is async).
        Returns 0 if no suitable victim found.

        Uses the cached health status (``_last_all_status``) rather than a fresh
        synchronous re-probe — this runs inside ensure_role message handling on
        the main loop, and an N×3s probe storm here was issue #8.  The cache is
        at most one health interval stale, which is fine for an eviction choice.
        """
        hp = self._router._healthy_ports
        if hp is None:
            return 0.0
        all_status = self._last_all_status or []
        active = [s for s in all_status if s["healthy"]]
        if len(active) <= 1:
            return 0.0

        top = self._router._find_top_model()
        top_port = top["port"] if top else None
        # Also protect the current conscious model
        conscious = getattr(self.brain, '_current_conscious', None)
        conscious_name = conscious.name if conscious else None

        now = time.monotonic()
        candidates = []
        for s in active:
            if s["port"] == top_port:
                continue
            if self._starting.get(s["name"], False):
                continue
            # Minimum residency: don't tear down a just-started model to make
            # room for another (that is the thrash we're preventing).
            if now - self._started_at.get(s["name"], 0.0) < self._MIN_RESIDENCY_S:
                continue
            # Never evict the model serving the current conscious stream
            # (we can't easily map stream→model, so protect top model only)
            candidates.append((self._eviction_score(s), s))

        if not candidates:
            return 0.0

        candidates.sort(key=lambda x: x[0])
        victim = candidates[0][1]
        freed = self._running_footprint_gb(victim)  # real RSS freed (KV incl.)

        self.add_to_log(
            f"Evicting {victim['name']} ({freed:.0f}GB) to make room "
            f"(need {need_gb:.0f}GB, avail {avail_gb:.1f}GB)"
        )
        # Record the eviction so ensure_role won't immediately re-request it.
        self._evicted_at[victim["name"]] = now
        self._try_stop(victim)
        return freed

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def _check_all_models(self) -> List[Dict]:
        """Check health of every model in the registry."""
        import urllib.request as _ur
        results = []
        for model in self._registry:
            port     = model["port"]
            pid_file = _TOOLS_DIR / f"llama-server-{port}.pid"
            pid      = 0
            running  = False
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                    running = True
                except (ValueError, OSError):
                    pid, running = 0, False

            healthy = False
            if running:
                try:
                    r = _ur.urlopen(
                        f"http://{self._host}:{port}/health", timeout=3
                    )
                    healthy = r.status == 200
                except Exception:
                    pass

            # Measure the model's real resident footprint (weights + KV cache +
            # runtime).  Track the observed peak per model so the start gate can
            # self-calibrate its footprint estimate after a model has run once.
            rss_gb = self._process_rss_gb(pid) if running else 0.0
            if rss_gb > 0:
                self._observed_rss_gb[model["name"]] = max(
                    self._observed_rss_gb.get(model["name"], 0.0), rss_gb
                )

            req_stats = self._router.get_stats_summary().get(model["name"], {})
            results.append({
                "name":         model["name"],
                "family":       model.get("family", ""),
                "port":         port,
                "pid":          pid,
                "running":      running,
                "healthy":      healthy,
                "size_gb":      model.get("size_gb", 0),
                "rss_gb":       round(rss_gb, 1),
                "parameters_b": model.get("parameters_b", 0),
                "prefill_tps":  model.get("prefill_tps", 0),
                "decode_tps":   model.get("decode_tps", 0),
                "roles":        model.get("roles", []),
                "default_for":  model.get("default_for", []),
                "vision":       model.get("vision", False),
                "script":       model.get("script", ""),
                "requests_total":  req_stats.get("requests_total", 0),
                "avg_latency_s":   req_stats.get("avg_latency_s", 0),
                "last_request":    req_stats.get("last_request"),
            })
        return results

    def _write_active_json(self, all_status: List[Dict]) -> None:
        """Write tools/llm-active.json for consumption by hardware sensor."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram_total_gb     = round(vm.total     / 1024 ** 3, 1)
            ram_available_gb = round(vm.available / 1024 ** 3, 1)
        except Exception:
            ram_total_gb = ram_available_gb = 0.0

        active    = [s for s in all_status if s["healthy"]]
        # Real loaded RAM = sum of measured RSS (KV included), falling back to
        # weights-only size only when RSS wasn't measurable.
        loaded_gb     = round(sum(self._running_footprint_gb(s) for s in active), 1)
        loaded_gb_wts = round(sum(s.get("size_gb", 0) for s in active), 1)

        doc = {
            "updated":          datetime.now(timezone.utc).isoformat(),
            "ram_total_gb":     ram_total_gb,
            "ram_available_gb": ram_available_gb,
            "loaded_gb":        loaded_gb,            # measured RSS (KV incl.)
            "loaded_weights_gb": loaded_gb_wts,       # weights-only, for comparison
            "headroom_gb":      round(ram_available_gb - loaded_gb, 1),
            "active_count":     len(active),
            "models":           all_status,
        }
        try:
            _ACTIVE_PATH.write_text(
                json.dumps(doc, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("LlmManagementStream: could not write llm-active.json: %s", exc)

    # ------------------------------------------------------------------
    # Auto-start
    # ------------------------------------------------------------------

    def _try_start(self, model_entry: Dict) -> None:
        """Non-blocking: launch the model's start script in a background thread."""
        name   = model_entry["name"]
        # Refuse to spawn new starts while paused (winding-down).
        if self._paused:
            return
        script = _TOOLS_DIR / model_entry["script"]
        if not script.is_file():
            log.warning("LlmManagementStream: start script not found: %s", script)
            return

        self._starting[name] = True
        # Record start time for the minimum-residency check (don't evict a
        # model we just started) and clear any stale eviction cooldown.
        self._started_at[name] = time.monotonic()
        self._evicted_at.pop(name, None)
        self.add_to_log(f"Auto-starting {name} via {model_entry['script']} …")
        # HLD: adenosine depletes on heavy actions like starting LLMs.
        adenosine = getattr(self.brain, 'adenosine', None)
        if adenosine is not None:
            adenosine.drain_activity("llm_start")
        port = model_entry["port"]

        def _run() -> None:
            try:
                env = {**os.environ, "LLM_PORT": str(port)}
                result = subprocess.run(
                    ["bash", str(script)],
                    capture_output=True, text=True, timeout=240, env=env,
                )
                raw = result.stdout.strip()
                try:
                    out = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    out = {"raw": raw[:200]}
                if out.get("status") == "ok":
                    self.add_to_log(
                        f"{name} started: url={out.get('url','?')}"
                    )
                    self._failures[name] = 0
                else:
                    self.add_to_log(f"{name} start script output: {out}")
            except subprocess.TimeoutExpired:
                self.add_to_log(f"{name} auto-start timed out (240 s)")
            except Exception as exc:
                log.warning("LlmManagementStream._try_start %s: %s", name, exc)
            finally:
                self._starting[name] = False

        t = threading.Thread(target=_run, name=f"llm_start_{name}", daemon=True)
        self._background_threads.append(t)
        t.start()

    # ------------------------------------------------------------------
    # Public API (called by other streams / brain)
    # ------------------------------------------------------------------

    def ensure_running(self, timeout: int = 240) -> bool:
        """
        Synchronously ensure the default chat LLM is healthy.
        Called once at wakeup before any stream makes an LLM request.
        """
        # Find the default chat model
        chat_model = next(
            (m for m in self._registry if "chat" in m.get("default_for", [])),
            self._registry[0] if self._registry else None,
        )
        if chat_model is None:
            log.warning("LlmManagementStream.ensure_running: empty registry")
            return False

        port   = chat_model["port"]
        script = _TOOLS_DIR / chat_model["script"]

        import urllib.request as _ur
        def _healthy() -> bool:
            try:
                r = _ur.urlopen(
                    f"http://{self._host}:{port}/health", timeout=4
                )
                return r.status == 200
            except Exception:
                return False

        if _healthy():
            status = {
                "name":    chat_model["name"],
                "healthy": True,
                "model_id": chat_model["file"],
                "url":     f"http://{self._host}:{port}",
                "pid":     0,
                "status":  "running",
            }
            self.brain._llm_status = status
            self._failures[chat_model["name"]] = 0
            self.add_to_log(
                f"LLM healthy at startup: model={chat_model['name']} "
                f"url=http://{self._host}:{port}"
            )
            # Seed router with all currently healthy ports so first-tick
            # LLM requests route correctly (before the first health check fires).
            healthy_ports: set = set()
            for m in self._registry:
                try:
                    r = _ur.urlopen(
                        f"http://{self._host}:{m['port']}/health", timeout=2
                    )
                    if r.status == 200:
                        healthy_ports.add(m["port"])
                except Exception:
                    pass
            self._router.update_healthy_ports(healthy_ports)
            log.debug(
                "LlmManagementStream.ensure_running: seeded healthy ports %s",
                healthy_ports,
            )
            return True

        if not script.is_file():
            self.add_to_log(f"LLM not running and start script not found: {script}")
            return False

        # Non-blocking cold start (P2-b: waking must not block for up to the
        # 240s start timeout — HLD's waking state is "very short").  Kick the
        # start on the same debounced background path the health loop uses and
        # return; the loop degrades gracefully until the model is up (the
        # scheduler returns model_unavailable, chat retries, the health check
        # confirms readiness).
        self.add_to_log(
            f"LLM not running at startup — starting via {chat_model['script']} "
            f"(non-blocking) …"
        )
        self._try_start(chat_model)
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return last known status dict for the default chat model (legacy)."""
        return dict(self.brain._llm_status) if hasattr(self.brain, "_llm_status") else {}

    def get_all_status(self) -> List[Dict]:
        """Return last known status list for all models."""
        return list(self._last_all_status)

    def start_model(self, script_name: str) -> None:
        """Start a specific model by tools/ script name (non-blocking)."""
        entry = next(
            (m for m in self._registry if m["script"] == script_name), None
        )
        if entry:
            self._failures[entry["name"]] = self._AUTO_START_THRESHOLD
            self._try_start(entry)

    def stop_model(self, name: str) -> bool:
        """Stop a running model by registry name (non-blocking).

        Returns True if a stop was initiated, False if the model was not
        found or is already being started/stopped.
        """
        entry = next(
            (m for m in self._registry if m["name"] == name), None
        )
        if entry is None:
            log.warning("LlmManagementStream.stop_model: unknown model %r", name)
            return False
        if self._starting.get(name, False):
            log.warning("LlmManagementStream.stop_model: %s is currently starting", name)
            return False
        self._try_stop(entry)
        return True

    # ------------------------------------------------------------------
    # Stop helper
    # ------------------------------------------------------------------

    _STOP_SCRIPT = _TOOLS_DIR / "llm-stop.sh"

    def _try_stop(self, model_entry: Dict) -> None:
        """Non-blocking: kill a running model via llm-stop.sh in a background thread."""
        name = model_entry["name"]
        port = model_entry["port"]

        # Refuse to spawn new stops while paused (winding-down).
        if self._paused:
            return

        if not self._STOP_SCRIPT.is_file():
            log.warning("LlmManagementStream: stop script not found: %s", self._STOP_SCRIPT)
            return

        self._starting[name] = True  # reuse flag to prevent concurrent start/stop
        # Shadow set so the per-tick memory guard treats this model as already
        # freed (until the next health check confirms it down) and doesn't
        # re-issue the same stop every tick.
        self._stopped.add(name)
        self.add_to_log(f"Stopping {name} (port {port}) …")
        # HLD: adenosine depletes on heavy actions like stopping LLMs.
        adenosine = getattr(self.brain, 'adenosine', None)
        if adenosine is not None:
            adenosine.drain_activity("llm_stop")

        def _run() -> None:
            try:
                env = {**os.environ, "LLM_PORT": str(port)}
                result = subprocess.run(
                    ["bash", str(self._STOP_SCRIPT)],
                    capture_output=True, text=True, timeout=30, env=env,
                )
                raw = result.stdout.strip()
                try:
                    out = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    out = {"raw": raw[:200]}
                if out.get("status") == "ok":
                    self.add_to_log(f"{name} stopped (port {port})")
                else:
                    self.add_to_log(f"{name} stop output: {out}")
            except subprocess.TimeoutExpired:
                self.add_to_log(f"{name} stop timed out (30 s)")
            except Exception as exc:
                log.warning("LlmManagementStream._try_stop %s: %s", name, exc)
            finally:
                self._starting[name] = False

        t = threading.Thread(target=_run, name=f"llm_stop_{name}", daemon=True)
        self._background_threads.append(t)
        t.start()

    # ------------------------------------------------------------------
    # Registry helper
    # ------------------------------------------------------------------

    @staticmethod
    def _load_registry() -> List[Dict]:
        try:
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("LlmManagementStream: could not load registry: %s", exc)
            # Minimal fallback so the system still works without registry
            return [{
                "name": "default",
                "file": "unknown",
                "size_gb": 0,
                "script": os.getenv("LLM_DEFAULT_MODEL", "llm-gemma4-26b.sh"),
                "port": int(os.getenv("LLM_PORT", "8080")),
                "context_k": 64,
                "quantization": "unknown",
                "roles": ["chat", "reasoning", "stm", "alignment", "fast",
                          "vision", "code", "codegen"],
                "default_for": ["chat", "reasoning", "stm", "alignment",
                                "fast", "vision", "code", "codegen"],
                "vision": False,
                "description": "Fallback entry — registry file not found.",
            }]

    # ------------------------------------------------------------------
    # Stream protocol helpers
    # ------------------------------------------------------------------

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
