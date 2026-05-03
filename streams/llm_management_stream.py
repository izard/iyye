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
        # Expose on brain for other streams
        brain._llm_request_log = self._request_log

    def update_healthy_ports(self, ports: set) -> None:
        """Called by LlmManagementStream after each health check tick."""
        self._healthy_ports = ports

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
        """
        if conscious:
            model = self._find_top_model()
        else:
            model = self._find_model(role, task=task)

        # If scoring picked an offline model, request a background start
        # and return a healthy fallback so the caller never gets a
        # connection error.  _try_start is non-blocking so the preferred
        # model won't be ready by the time this method returns.
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

        if model is None:
            log.warning("LLMRouter: no registry model for role=%r — using env defaults", role)
            from llm_client import LLMClient
            return LLMClient(**kwargs)

        port    = model["port"]
        host    = os.environ.get("LLM_HOST", "127.0.0.1")
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

    def _effective_decode_tps(self, model: Dict) -> float:
        """Return decode tokens/s: registry value if set, else observed, else 0.

        HLD: each LLM has three performance properties — weights size,
        prefill tokens/second, decode tokens/second.  When registry values
        haven't been filled in yet (0), fall back to observed throughput
        from actual requests tracked by _record().
        """
        reg = model.get("decode_tps", 0)
        if reg > 0:
            return float(reg)
        stats = self._model_stats.get(model.get("name", ""))
        if stats:
            return stats.get("observed_decode_tps", 0.0)
        return 0.0

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
        role_importance = self._ROLE_IMPORTANCE.get(role, 0.5)
        task_value = urgency * quality_need * role_importance
        quality_score = task_value * model_quality  # 0..1 range

        # --- Latency ---
        prefill_tps = model.get("prefill_tps", 0)
        decode_tps  = self._effective_decode_tps(model)
        # Estimate time; use conservative defaults for unknowns
        if prefill_tps > 0:
            prefill_s = prompt_tok / prefill_tps
        else:
            prefill_s = prompt_tok / 50.0  # conservative 50 tok/s default
        if decode_tps > 0:
            decode_s = output_tok / decode_tps
        else:
            decode_s = output_tok / 15.0  # conservative 15 tok/s default

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
        """Post an ``ensure_role`` message to LlmManagementStream."""
        brain = self._brain
        if brain is None:
            return
        brain.post_message("llm_management", {
            "action": "ensure_role",
            "role": role,
            "model_name": model["name"],
            "task": task,
            "reason": f"get_client(role={role!r}) — best model {model['name']} not healthy",
        })
        log.info(
            "LLMRouter: requested ensure_role for %s (role=%r)",
            model["name"], role,
        )

    def _find_top_model(self) -> Optional[Dict]:
        """Return the most powerful healthy non-vision model.

        Primary key: parameters_b (quality).  Tiebreaker: decode throughput.
        """
        hp = self._healthy_ports
        candidates = [
            m for m in self._registry
            if not m.get("vision") and (hp is None or m["port"] in hp)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda m: (
            m.get("parameters_b", 0), self._effective_decode_tps(m),
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
        # Exclude the top model so it stays reserved for the conscious stream.
        top = self._find_top_model()
        top_port = top["port"] if top else None

        def _ok(m: Dict) -> bool:
            return m["port"] != top_port

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
            candidates = [m for m in self._registry
                          if role in m.get("roles", []) and _ok(m)]
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
        self._request_log.append(entry)

        s = self._model_stats.setdefault(model_name, {
            "requests_total": 0,
            "total_latency_s": 0.0,
            "last_request": None,
            "observed_decode_tps": 0.0,
        })
        s["requests_total"]  += 1
        s["total_latency_s"] += latency_s
        s["last_request"]     = entry["timestamp"]

        # Track observed decode throughput (~4 chars/token as rough estimate).
        if response_chars > 20 and latency_s > 0.1:
            observed = (response_chars / 4.0) / latency_s
            prev = s.get("observed_decode_tps", 0.0)
            # Exponential moving average (α=0.3) for smoothing.
            s["observed_decode_tps"] = (
                observed if prev == 0 else 0.3 * observed + 0.7 * prev
            )

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

        # Per-model failure counters and start-in-progress flags
        self._failures:    Dict[str, int]  = {}
        self._starting:    Dict[str, bool] = {}

        self._ticks_since_check: int = self.HEALTH_INTERVAL  # check on first tick
        self._last_all_status:   List[Dict] = []

        self._host        = os.getenv("LLM_HOST", "127.0.0.1")
        self._auto_start  = bool(os.getenv("LLM_AUTO_START", ""))
        self._vision_auto = bool(os.getenv("LLM_VISION_AUTO_START", ""))

        # Minimum available RAM (GB) before shedding a low-value model.
        self._HEADROOM_MIN_GB = 4.0

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
        # Process inter-stream messages (e.g. restart requests from self-reflection).
        for msg in self.brain.drain_messages("llm_management"):
            self._handle_message(msg)

        self._ticks_since_check += 1
        if self._ticks_since_check < self.HEALTH_INTERVAL:
            return None
        self._ticks_since_check = 0

        all_status = self._check_all_models()
        self._last_all_status = all_status

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
                    and self._failures[name] >= self._AUTO_START_THRESHOLD
                    and ("chat" in s.get("default_for", [])
                         or (self._vision_auto and s.get("vision")))
                )
                if should_auto:
                    self._try_start(s)

        # Proactive resource management: shed a low-value model when RAM is low.
        self._manage_resources(all_status)

        self.checkpoint()
        return {"llm_status": all_status}

    # ------------------------------------------------------------------
    # Inter-stream message handling
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "restart":
            role = msg.get("role", "chat")
            model = self._router._find_model(role)
            if model and not self._starting.get(model["name"], False):
                self.add_to_log(
                    f"Restart requested for role={role!r} ({msg.get('reason','?')})"
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
        if self._starting.get(model_name, False):
            log.debug("ensure_role: %s already starting", model_name)
            return
        # Already healthy — nothing to do
        hp = self._router._healthy_ports
        if hp is not None and entry["port"] in hp:
            return

        # Check RAM headroom
        ram_needed = entry.get("ram_required_gb", entry.get("size_gb", 20))
        try:
            import psutil
            ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            ram_avail = 32.0  # optimistic default
        headroom_after = ram_avail - ram_needed

        # Decide: start if we'd still have ≥2 GB headroom, or if this is a
        # small model (< 10 GB) that serves a high-frequency role.
        is_small = ram_needed < 10
        if headroom_after < 2.0 and not is_small:
            # Try eviction first
            freed = self._try_evict_for(ram_needed, ram_avail)
            if freed < ram_needed * 0.5:
                log.info(
                    "ensure_role: skipping %s — RAM too tight "
                    "(need %.1f GB, avail %.1f GB, freed %.1f GB)",
                    model_name, ram_needed, ram_avail, freed,
                )
                self.add_to_log(
                    f"ensure_role: declined {model_name} for {role} — "
                    f"RAM pressure ({ram_avail:.0f} GB avail, need {ram_needed:.0f} GB)"
                )
                return

        self.add_to_log(
            f"ensure_role: starting {model_name} for {role} ({reason})"
        )
        self._try_start(entry)

    # ------------------------------------------------------------------
    # Proactive resource management
    # ------------------------------------------------------------------

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
        size_gb = status.get("size_gb", 0)
        return (defaults * 5.0
                + min(total_req, 50) * 0.1  # cap contribution at 5 pts
                + recency_bonus
                - size_gb * 0.05)  # small bonus for freeing larger models

    def _manage_resources(self, all_status: List[Dict]) -> None:
        """Stop the least valuable model when available RAM drops below threshold.

        Never stops the top model (reserved for conscious stream) or a model
        that is currently being started/stopped.  Among remaining candidates
        the one with the lowest eviction score is stopped first.
        """
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / 1024 ** 3
        except Exception:
            return

        if available_gb >= self._HEADROOM_MIN_GB:
            return

        active = [s for s in all_status if s["healthy"]]
        if len(active) <= 1:
            return  # never stop the last running model

        top = self._router._find_top_model()
        top_port = top["port"] if top else None

        candidates = []
        for s in active:
            if s["port"] == top_port:
                continue
            if self._starting.get(s["name"], False):
                continue
            candidates.append((self._eviction_score(s), s))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        victim = candidates[0][1]

        self.add_to_log(
            f"RAM low ({available_gb:.1f}GB free) — stopping "
            f"{victim['name']} ({victim['size_gb']}GB) to free resources"
        )
        self._try_stop(victim)

    def _try_evict_for(self, need_gb: float, avail_gb: float) -> float:
        """Try to free *need_gb* of RAM by evicting the least valuable model.

        Returns the amount of RAM (GB) that the evicted model will free
        (actual RAM won't be available immediately — model shutdown is async).
        Returns 0 if no suitable victim found.
        """
        hp = self._router._healthy_ports
        if hp is None:
            return 0.0
        all_status = self._check_all_models()
        active = [s for s in all_status if s["healthy"]]
        if len(active) <= 1:
            return 0.0

        top = self._router._find_top_model()
        top_port = top["port"] if top else None
        # Also protect the current conscious model
        conscious = getattr(self.brain, '_current_conscious', None)
        conscious_name = conscious.name if conscious else None

        candidates = []
        for s in active:
            if s["port"] == top_port:
                continue
            if self._starting.get(s["name"], False):
                continue
            # Never evict the model serving the current conscious stream
            # (we can't easily map stream→model, so protect top model only)
            candidates.append((self._eviction_score(s), s))

        if not candidates:
            return 0.0

        candidates.sort(key=lambda x: x[0])
        victim = candidates[0][1]
        freed = victim.get("size_gb", 0)

        self.add_to_log(
            f"Evicting {victim['name']} ({freed:.0f}GB) to make room "
            f"(need {need_gb:.0f}GB, avail {avail_gb:.1f}GB)"
        )
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

            req_stats = self._router.get_stats_summary().get(model["name"], {})
            results.append({
                "name":         model["name"],
                "family":       model.get("family", ""),
                "port":         port,
                "pid":          pid,
                "running":      running,
                "healthy":      healthy,
                "size_gb":      model.get("size_gb", 0),
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
        loaded_gb = round(sum(s["size_gb"] for s in active), 1)

        doc = {
            "updated":          datetime.now(timezone.utc).isoformat(),
            "ram_total_gb":     ram_total_gb,
            "ram_available_gb": ram_available_gb,
            "loaded_gb":        loaded_gb,
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
        script = _TOOLS_DIR / model_entry["script"]
        if not script.is_file():
            log.warning("LlmManagementStream: start script not found: %s", script)
            return

        self._starting[name] = True
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

        threading.Thread(target=_run, name=f"llm_start_{name}", daemon=True).start()

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

        self.add_to_log(
            f"LLM not running at startup — starting via {chat_model['script']} …"
        )
        try:
            env = {**os.environ, "LLM_PORT": str(port)}
            result = subprocess.run(
                ["bash", str(script)],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            raw = result.stdout.strip()
            try:
                out = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                out = {}
            if out.get("status") == "ok":
                self.add_to_log(
                    f"LLM started: model={chat_model['name']} "
                    f"url={out.get('url','?')}"
                )
                self._failures[chat_model["name"]] = 0
                return _healthy()
            else:
                self.add_to_log(f"LLM start script output: {out}")
                return False
        except subprocess.TimeoutExpired:
            self.add_to_log(f"LLM start timed out after {timeout}s")
            return False
        except Exception as exc:
            self.add_to_log(f"LLM start error: {exc}")
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

        if not self._STOP_SCRIPT.is_file():
            log.warning("LlmManagementStream: stop script not found: %s", self._STOP_SCRIPT)
            return

        self._starting[name] = True  # reuse flag to prevent concurrent start/stop
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

        threading.Thread(target=_run, name=f"llm_stop_{name}", daemon=True).start()

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
