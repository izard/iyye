# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bounded asynchronous LLM job scheduler (HLD issue #3).

The brain's main loop is single-threaded; every synchronous LLM call blocks it
(a chat decode can stall the whole cognitive loop for minutes).  This scheduler
moves the blocking I/O onto worker threads while keeping the cooperative,
single-threaded model intact:

    * Streams submit an immutable :class:`LLMJob` and return immediately.
    * Worker threads (one or more per llama.cpp port) perform the blocking call.
    * **Results are applied on the main thread** — a stream calls
      :meth:`LLMScheduler.poll` at the top of its ``execute()`` and mutates its
      own state there.  No locks ever leak into stream bodies.

Two invariants do most of the work:

    1. Results applied on the main thread  → stream state stays single-threaded.
    2. **At most one in-flight job per stream** → per-contact response ordering
       (each contact already has its own UserChatStream) and chat backpressure
       (the user's next message waits in the stream's own queue) fall out for
       free; no dedicated ordering mechanism is needed.

Model resolution is done at *submit* time to pick a port queue, and re-checked
by the worker just before the call so a model evicted by memory-pressure
(issue #5) between submit and execution fails fast with ``model_unavailable``
rather than hanging — the owning stream simply re-submits.

The scheduler composes with :class:`LLMRouter` rather than duplicating it: the
router decides *which* model (and reserves the top model for conscious
streams); the scheduler decides *when / in what order* and caps per-port
concurrency (llama.cpp is effectively single-slot — default limit 1).
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

log = logging.getLogger("Iyye.LLMScheduler")

# Sentinel port for "router returned no registry model" (env-default client).
_ENV_PORT = -1

# kind -> priority (lower number served first within a port's queue).
_KIND_PRIORITY: Dict[str, int] = {
    "chat_conscious":    0,
    "chat_subconscious": 1,
    "reflection":        1,   # conscious self-reflection report
    "research":          1,   # user-requested web-research rephrase
    "stm":               2,
    "profile":           2,
    "alignment":         2,   # background stream-alignment scoring
    "codegen":           3,
    "refine":            3,
    "planned":           3,   # planned-continuation step execution
    "replay":            4,
}
_DEFAULT_PRIORITY = 3


# ----------------------------------------------------------------------
# Immutable job / call / result records (safe to cross the thread boundary)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class LLMCall:
    """What to ask the model — exactly one form is populated."""
    prompt_name: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    @classmethod
    def from_file(cls, prompt_name: str, **variables: Any) -> "LLMCall":
        """A ``client.complete_from_file(prompt_name, **variables)`` call."""
        return cls(prompt_name=prompt_name, variables=dict(variables))

    @classmethod
    def prompt(cls, user_prompt: str, system_prompt: Optional[str] = None) -> "LLMCall":
        """A ``client.complete(user_prompt, system_prompt)`` call."""
        return cls(user_prompt=user_prompt, system_prompt=system_prompt)


@dataclass(frozen=True)
class LLMJob:
    job_id: str
    stream_name: str
    role: str
    kind: str
    call: LLMCall
    cycle_id: int
    conscious: bool = False
    client_kwargs: Dict[str, Any] = field(default_factory=dict)
    task: Optional[Dict[str, Any]] = None
    deadline_s: Optional[float] = None
    priority: int = _DEFAULT_PRIORITY
    submitted_monotonic: float = 0.0


@dataclass(frozen=True)
class LLMResult:
    job_id: str
    stream_name: str
    ok: bool
    text: str = ""
    error: Optional[str] = None          # model_unavailable | timeout | stale_cycle | cancelled | deadline_exceeded | exception:...
    model_name: Optional[str] = None
    latency_s: float = 0.0
    cycle_id: int = 0
    discarded: bool = False              # stale: paused/wrong-cycle/cancelled — stream should no-op


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

class LLMScheduler:
    """Per-port, priority-ordered async executor for LLM jobs.

    Public API splits cleanly into a **main-thread** surface (submit / poll /
    has_inflight / cancel_stream) and a **lifecycle** surface driven by the
    brain (on_wake / begin_pause / settle / close).
    """

    def __init__(
        self,
        router: Any,
        brain: Any = None,
        per_port_limit: int = 1,
    ) -> None:
        self._router = router
        self._brain = brain
        self._per_port_limit = max(1, int(per_port_limit))

        self._lock = threading.Lock()
        self._seq = 0
        self._cycle_id = 0
        self._accepting = True
        self._running = True

        # stream_name -> job_id of its single in-flight job
        self._inflight: Dict[str, str] = {}
        # stream_name -> completed result awaiting poll()
        self._results: Dict[str, LLMResult] = {}
        # job_ids marked stale by cancel_stream
        self._stale: set = set()

        # port -> intake PriorityQueue; port -> worker threads
        self._port_queues: Dict[int, "queue.PriorityQueue"] = {}
        self._port_workers: Dict[int, List[threading.Thread]] = {}

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def submit_call(
        self,
        *,
        stream_name: str,
        role: str,
        kind: str,
        call: LLMCall,
        conscious: bool = False,
        client_kwargs: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None,
        deadline_s: Optional[float] = None,
    ) -> bool:
        """Build and enqueue a job for *stream_name*.

        Returns False (rejected, no work lost) when the scheduler is paused or
        the stream already has an in-flight job (the 1-in-flight rule)."""
        with self._lock:
            if not self._accepting or stream_name in self._inflight:
                return False

        # Resolve the model to choose a port queue.  Cheap (no I/O); may post an
        # ensure_role mailbox message — fine, this runs on the main thread.
        try:
            model = self._router.resolve(role, conscious=conscious, task=task)
        except Exception as exc:
            log.warning("scheduler: model resolution failed for role=%r: %s", role, exc)
            model = None
        port = model.get("port") if model else _ENV_PORT
        priority = _KIND_PRIORITY.get(kind, _DEFAULT_PRIORITY)

        with self._lock:
            if not self._accepting or stream_name in self._inflight:
                return False
            self._seq += 1
            job = LLMJob(
                job_id=uuid4().hex,
                stream_name=stream_name,
                role=role,
                kind=kind,
                call=call,
                cycle_id=self._cycle_id,
                conscious=conscious,
                client_kwargs=dict(client_kwargs or {}),
                task=task,
                deadline_s=deadline_s,
                priority=priority,
                submitted_monotonic=time.monotonic(),
            )
            # Drop any unpolled result still sitting under this name: it belongs
            # to a prior job/instance that was never consumed (e.g. a same-named
            # stream that retired with a result pending).  A fresh submission
            # supersedes it, so it must not be mistaken for this job's result
            # (issue #6).
            self._results.pop(stream_name, None)
            self._inflight[stream_name] = job.job_id
            q = self._ensure_port_worker_locked(port)
            q.put((priority, self._seq, job, model))
        # Shadow-journal the submit (Phase 0 causal recording).  The prompt is
        # identified compactly — its template name, or a digest of the inline
        # prompt — so replay can match this submit to its result without
        # storing the full rendered context on every job.
        self._journal_submit(job)
        return True

    def _journal(self):
        return getattr(self._brain, "journal", None) if self._brain is not None else None

    def _journal_submit(self, job: "LLMJob") -> None:
        from event_journal import emit, fingerprint
        call = job.call
        prompt_desc = call.prompt_name or f"inline:{fingerprint(call.user_prompt or '')}"
        # Record which prompt *version* served this job (default "base") so the
        # sleep self-improvement pass can attribute the outcome to the version
        # that produced it — the per-version reward signal for #6 prompt tuning.
        prompt_version = "base"
        if call.prompt_name:
            try:
                from llm_client import active_prompt_version
                prompt_version = active_prompt_version(call.prompt_name)
            except Exception:
                pass
        emit(self._journal(), "llm_submit",
             job_id=job.job_id, stream=job.stream_name, kind=job.kind,
             role=job.role, conscious=job.conscious, prompt=prompt_desc,
             prompt_name=call.prompt_name or None, prompt_version=prompt_version)

    def poll(self, stream_name: str) -> Optional[LLMResult]:
        """Return (and clear) a completed result for *stream_name*, or None.

        A result produced in a cycle that has since ended is returned with
        ``discarded=True`` so the stream can no-op cleanly."""
        with self._lock:
            res = self._results.pop(stream_name, None)
            if res is not None and not res.discarded and res.cycle_id != self._cycle_id:
                res = dataclasses.replace(res, discarded=True)
        return res

    def has_inflight(self, stream_name: str) -> bool:
        with self._lock:
            return stream_name in self._inflight

    def cancel_stream(self, stream_name: str) -> None:
        """Mark *stream_name*'s in-flight job stale and drop any landed result.

        Best-effort: a running HTTP call is not truly preempted; its result is
        discarded when the worker returns."""
        with self._lock:
            jid = self._inflight.get(stream_name)
            if jid:
                self._stale.add(jid)
            self._results.pop(stream_name, None)

    # ------------------------------------------------------------------
    # Lifecycle (brain-driven)
    # ------------------------------------------------------------------

    def on_wake(self, cycle_id: int) -> None:
        """Start of a new awake cycle: adopt *cycle_id*, resume accepting, and
        drop any results left over from an older cycle."""
        with self._lock:
            self._cycle_id = cycle_id
            self._accepting = True
            self._results = {
                k: v for k, v in self._results.items() if v.cycle_id == cycle_id
            }

    def begin_pause(self) -> None:
        """Wind-down: stop accepting new submits.  In-flight workers continue;
        their results are dropped by the cycle check on the next wake."""
        with self._lock:
            self._accepting = False

    def settle(self, timeout_s: float) -> bool:
        """Wait up to *timeout_s* for in-flight jobs to drain.  Returns True if
        drained, False on timeout (workers keep running; results discarded)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._inflight:
                    return True
            time.sleep(0.02)
        with self._lock:
            return not self._inflight

    def close(self) -> None:
        """Stop worker loops (daemon threads exit with the process anyway)."""
        self._running = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_port_worker_locked(self, port: int) -> "queue.PriorityQueue":
        """Return the intake queue for *port*, spawning its worker(s) on first
        use.  Caller must hold ``self._lock``."""
        q = self._port_queues.get(port)
        if q is not None:
            return q
        q = queue.PriorityQueue()
        self._port_queues[port] = q
        workers: List[threading.Thread] = []
        for i in range(self._per_port_limit):
            t = threading.Thread(
                target=self._worker, args=(port, q),
                name=f"llm_sched_p{port}_{i}", daemon=True,
            )
            workers.append(t)
            t.start()
        self._port_workers[port] = workers
        return q

    def _worker(self, port: int, q: "queue.PriorityQueue") -> None:
        while self._running:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            job = None
            try:
                _priority, _seq, job, model = item
                res = self._run_job(job, model)
            except Exception as exc:  # never let a worker thread die silently
                log.warning("scheduler: worker crash on %s: %s",
                            getattr(job, "stream_name", "?"), exc)
                # Synthesize an error result so the owning stream is ALWAYS
                # unwedged.  Without this, a crash here left _inflight set
                # forever and the stream could never poll or submit again
                # (issue #6).
                res = (
                    LLMResult(job_id=job.job_id, stream_name=job.stream_name,
                              ok=False, error=f"worker_crash:{type(exc).__name__}",
                              cycle_id=job.cycle_id)
                    if job is not None else None
                )
            finally:
                q.task_done()
            # Always clear the in-flight slot and deliver a result (real or
            # synthesized) so the stream never wedges.
            if res is not None:
                with self._lock:
                    self._inflight.pop(res.stream_name, None)
                    self._results[res.stream_name] = res
                    self._stale.discard(res.job_id)
                # Shadow-journal the resolved result (Phase 0).  Recorded at
                # delivery, not poll, so every result — including ones a
                # retired/pruned stream never consumes — is captured: that is
                # exactly the "computed but never delivered" case the prune
                # race produced.  Thread-safe; journal append is locked.
                self._journal_result(res)
            elif job is not None:
                # Couldn't even build a result — still free the slot.
                with self._lock:
                    self._inflight.pop(job.stream_name, None)

    def _journal_result(self, res: "LLMResult") -> None:
        from event_journal import emit, clip
        emit(self._journal(), "llm_result",
             job_id=res.job_id, stream=res.stream_name, ok=res.ok,
             error=res.error, model=res.model_name, latency_s=res.latency_s,
             discarded=res.discarded, text=clip(res.text))

    def _run_job(self, job: LLMJob, model: Optional[Dict[str, Any]]) -> LLMResult:
        def _result(**kw) -> LLMResult:
            base = dict(
                job_id=job.job_id, stream_name=job.stream_name,
                cycle_id=job.cycle_id,
                model_name=(model.get("name") if model else None),
            )
            base.update(kw)
            return LLMResult(**base)

        with self._lock:
            stale = job.job_id in self._stale
            cur_cycle = self._cycle_id
        if stale:
            return _result(ok=False, error="cancelled", discarded=True)
        if job.cycle_id != cur_cycle:
            return _result(ok=False, error="stale_cycle", discarded=True)
        if (job.deadline_s is not None
                and (time.monotonic() - job.submitted_monotonic) > job.deadline_s):
            return _result(ok=False, error="deadline_exceeded", discarded=True)

        # Eviction re-check: the model may have been stopped since submit.
        hp = getattr(self._router, "_healthy_ports", None)
        if model is not None and hp is not None and model.get("port") not in hp:
            return _result(ok=False, error="model_unavailable")

        try:
            if model is None:
                client = self._router.get_client(
                    job.role, conscious=job.conscious, task=job.task,
                    **job.client_kwargs,
                )
            else:
                client = self._router.build_client(model, job.role, **job.client_kwargs)

            t0 = time.monotonic()
            c = job.call
            if c.prompt_name is not None:
                text = client.complete_from_file(c.prompt_name, **c.variables)
            else:
                text = client.complete(c.user_prompt, c.system_prompt)
            latency = time.monotonic() - t0
            return LLMResult(
                job_id=job.job_id, stream_name=job.stream_name, ok=True,
                text=text or "", cycle_id=job.cycle_id, latency_s=latency,
                model_name=getattr(client, "_model_name",
                                   model.get("name") if model else None),
            )
        except Exception as exc:
            low = f"{type(exc).__name__}: {exc}".lower()
            if "timeout" in low or "timed out" in low:
                return _result(ok=False, error="timeout")
            # A connection failure against a port the router believed healthy
            # means the model died (e.g. OOM-killed) but the 30-tick health
            # check hasn't caught up.  Evict the port so the next request
            # reroutes, and surface a retry-friendly error instead of a raw
            # exception (the model is likely restarting).
            if model is not None and self._looks_like_connection_error(low):
                try:
                    self._router.mark_port_unhealthy(model.get("port"))
                except Exception:
                    pass
                log.warning("scheduler: connection failure on %s (port %s) — "
                            "evicting port, returning model_unavailable: %s",
                            model.get("name"), model.get("port"), exc)
                return _result(ok=False, error="model_unavailable")
            return _result(ok=False, error=f"exception:{type(exc).__name__}:{exc}")

    @staticmethod
    def _looks_like_connection_error(low: str) -> bool:
        """True for a transport-level "can't reach the server" failure (refused
        connection, reset, DNS/socket error) — distinct from an HTTP error the
        server itself returned."""
        return any(s in low for s in (
            "connection", "connect", "refused", "econnrefused",
            "max retries", "remotedisconnected", "reset by peer",
            "broken pipe", "newconnectionerror", "failed to establish",
        ))


# ----------------------------------------------------------------------
# Stream mixin — uniform poll → apply / busy → wait / idle → submit shape
# ----------------------------------------------------------------------

class LLMConsumerMixin:
    """Mixin for streams that issue LLM calls through the scheduler.

    Usage in ``execute()``::

        result = self._llm_poll()
        if result is not None:
            return self._on_llm_result(result, ctx)   # apply on main thread
        if self._llm_busy():
            return None                                # still waiting
        # idle — maybe start new work
        self._llm_submit(role="chat", kind="chat_conscious",
                         call=LLMCall.from_file("chat_response", **vars))

    The consuming stream stores whatever turn-context it needs to apply the
    result in a single instance slot (e.g. ``self._pending_turn``); because of
    the one-in-flight rule there is never more than one outstanding context.
    """

    def _llm_scheduler(self):
        return getattr(getattr(self, "brain", None), "llm_scheduler", None)

    def _llm_busy(self) -> bool:
        sch = self._llm_scheduler()
        return bool(sch and sch.has_inflight(self.name))

    def _llm_poll(self) -> Optional[LLMResult]:
        sch = self._llm_scheduler()
        return sch.poll(self.name) if sch is not None else None

    def _llm_submit(
        self,
        *,
        role: str,
        kind: str,
        call: LLMCall,
        conscious: bool = False,
        client_kwargs: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None,
        deadline_s: Optional[float] = None,
    ) -> bool:
        sch = self._llm_scheduler()
        if sch is None:
            return False
        return sch.submit_call(
            stream_name=self.name, role=role, kind=kind, call=call,
            conscious=conscious, client_kwargs=client_kwargs, task=task,
            deadline_s=deadline_s,
        )

    def _llm_cancel(self) -> None:
        sch = self._llm_scheduler()
        if sch is not None:
            sch.cancel_stream(self.name)


__all__ = [
    "LLMCall", "LLMJob", "LLMResult", "LLMScheduler", "LLMConsumerMixin",
]
