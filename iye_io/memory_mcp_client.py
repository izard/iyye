#!/usr/bin/env python3
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
"""
Long-term memory client backed by LanceDB.

Stores and retrieves facts using full-text search (FTS) and vector
similarity search.  Before inserting, store_fact checks for a
semantically near-duplicate (cosine similarity ≥ DEDUP_SIMILARITY)
and, if one is found, bumps its confidence instead of inserting a
new row.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Memory")

_TABLE = "facts"
EMBED_DIM = 384          # all-MiniLM-L6-v2
_MODEL_DIR = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
DEDUP_DISTANCE = 0.10    # cosine distance threshold (≈ similarity ≥ 0.90)
CONFIDENCE_BUMP = 0.05   # added to existing fact confidence when duplicate seen
_MAX_PROVENANCE = 500    # hard cap on provenance string length

# Durability rank: higher = survives longer.  Matches STM's _TF_RANK so
# dedup in both stores applies the same upgrade logic.
_TF_RANK = {tf: i for i, tf in enumerate(
    ("ephemeral", "session", "today", "recent", "dated", "permanent")
)}

# ISO-like timestamp tail — used to normalise provenance entries for dedup.
_TS_TAIL_RE = re.compile(r"\s+at\s+\d{4}-\d{2}-\d{2}T[\d:.+Z-]+$")


def _merge_provenance(old_prov: str, new_prov: str, cap: int = _MAX_PROVENANCE) -> str:
    """Merge provenance strings, avoiding duplicates from the same source.

    Each comma-separated segment is normalised by stripping trailing ISO
    timestamps (``... at 2026-04-27T08:05:11+00:00``) before comparison
    so that repeated sleep-replay entries from the same stream are not
    appended endlessly.
    """
    if not new_prov:
        return old_prov
    if not old_prov:
        return new_prov
    # Normalise: strip trailing timestamps so "sleep replay from X at T1"
    # and "sleep replay from X at T2" both reduce to "sleep replay from X".
    existing_keys = {
        _TS_TAIL_RE.sub("", seg.strip())
        for seg in old_prov.split(",") if seg.strip()
    }
    new_key = _TS_TAIL_RE.sub("", new_prov.strip())
    if new_key in existing_keys:
        return old_prov          # source already represented
    merged = f"{old_prov}, {new_prov}"
    if len(merged) > cap:
        return old_prov          # cap reached — don't grow further
    return merged

# HLD: "Each fact has a tag, which includes date and time, confidence level (0.2-1.0),
# provenance (person, agent, subsystem), time frame when fact is true
# (e.g. always, today, 2 weeks ago, etc), and text describing the inputs
# that contributed to fact inference."
_SCHEMA = pa.schema([
    pa.field("id",            pa.string()),
    pa.field("text",          pa.string()),
    pa.field("confidence",    pa.float32()),
    pa.field("source",        pa.string()),
    pa.field("provenance",    pa.string()),
    pa.field("time_frame",    pa.string()),   # HLD: "time frame when fact is true"
    pa.field("timestamp",     pa.string()),
    pa.field("metadata_json", pa.string()),
    pa.field("media_path",    pa.string()),   # HLD: "tgz file that may contain supporting media"
    pa.field("vector",        pa.list_(pa.float32(), EMBED_DIM)),
])


class MemoryClient:
    """
    LanceDB-backed long-term memory with semantic deduplication.

    Usage:
        mem = MemoryClient("iyye_memory")
        fid = mem.store_fact("The CPU was at 80% during the planning session")
        results = mem.search_text("CPU usage")
        recent = mem.get_recent_facts(50)
    """

    def __init__(self, db_path: str = "iyye_memory") -> None:
        self._db = lancedb.connect(str(PROJECT_ROOT / db_path))
        self._embedder = None  # lazy-loaded SentenceTransformer
        # Open existing table or create fresh one with full schema.
        # Wraps both steps in a try/except so a corrupted or missing table
        # is automatically recreated rather than crashing the brain on startup.
        try:
            if _TABLE in self._db.table_names():
                self._tbl = self._db.open_table(_TABLE)
            else:
                self._tbl = self._db.create_table(_TABLE, schema=_SCHEMA)
        except Exception as exc:
            log.warning("LanceDB open failed (%s) — recreating table", exc)
            try:
                self._db.drop_table(_TABLE)
            except Exception:
                pass
            self._tbl = self._db.create_table(_TABLE, schema=_SCHEMA)
        self._fts_dirty = True
        self._migrate_schema()
        log.info("MemoryClient ready at %s", db_path)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            if not _MODEL_DIR.exists():
                log.warning("Embedder model not found at %s — run bin/download-embedder.sh", _MODEL_DIR)
                return None
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    str(_MODEL_DIR), local_files_only=True,
                )
                log.info("Embedder loaded from %s (dim=%d)", _MODEL_DIR, EMBED_DIM)
            except Exception as exc:
                log.warning("Could not load embedder, dedup disabled: %s", exc)
        return self._embedder

    def _embed(self, text: str) -> Optional[List[float]]:
        embedder = self._get_embedder()
        if embedder is None:
            return None
        try:
            return embedder.encode(text, show_progress_bar=False).tolist()
        except Exception as exc:
            log.warning("Embedding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def _migrate_schema(self) -> None:
        """
        Ensure the table has all required columns.

        If the 'vector' column is missing (pre-embedding schema), the
        entire table is recreated with embeddings generated for every
        existing row so that semantic dedup works immediately.
        """
        existing_cols = {f.name for f in self._tbl.schema}

        # Scalar column migrations (cheap, in-place)
        scalar_migrations = {
            "confidence": ("add_column", pa.field("confidence", pa.float32()), 0.5),
            "source":     ("add_column", pa.field("source",     pa.string()),  "agent"),
            "provenance": ("add_column", pa.field("provenance", pa.string()),  ""),
            "time_frame": ("add_column", pa.field("time_frame", pa.string()),  "permanent"),
            "media_path": ("add_column", pa.field("media_path", pa.string()),  ""),
        }
        for col, (_, field, default) in scalar_migrations.items():
            if col not in existing_cols:
                try:
                    self._tbl.add_columns({col: str(default)})
                    log.info("Migrated memory schema: added column '%s'", col)
                    existing_cols.add(col)
                except Exception as exc:
                    log.warning("Schema migration failed for '%s': %s", col, exc)

        # Vector column migration — requires table recreation with backfill
        if "vector" not in existing_cols:
            self._migrate_add_vectors()

    def _migrate_add_vectors(self) -> None:
        """Recreate the facts table with embeddings for all existing rows."""
        log.info("Migrating memory table to include vector embeddings…")
        try:
            rows = self._tbl.to_pandas().to_dict("records")
            embedder = self._get_embedder()

            enriched = []
            for row in rows:
                text = row.get("text", "")
                if embedder and text:
                    try:
                        vec = embedder.encode(text, show_progress_bar=False).tolist()
                    except Exception:
                        vec = [0.0] * EMBED_DIM
                else:
                    vec = [0.0] * EMBED_DIM
                enriched.append({
                    "id":            row.get("id", str(uuid.uuid4())),
                    "text":          text,
                    "confidence":    float(row.get("confidence") or 0.5),
                    "source":        row.get("source") or "agent",
                    "provenance":    row.get("provenance") or "",
                    "time_frame":    row.get("time_frame") or "permanent",
                    "timestamp":     row.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    "metadata_json": row.get("metadata_json") or "{}",
                    "media_path":    row.get("media_path") or "",
                    "vector":        vec,
                })

            self._db.drop_table(_TABLE)
            self._tbl = self._db.create_table(_TABLE, data=enriched, schema=_SCHEMA)
            self._fts_dirty = True
            log.info("Vector migration complete (%d rows backfilled)", len(enriched))
        except Exception as exc:
            log.error("Vector migration failed, recreating empty table: %s", exc)
            self._tbl = self._db.create_table(_TABLE, schema=_SCHEMA, exist_ok=True)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _find_duplicate(self, embedding: List[float]) -> Optional[Tuple[str, float]]:
        """
        Search for a semantically equivalent existing fact.

        Returns (fact_id, existing_confidence) if a near-duplicate is
        found (cosine distance < DEDUP_DISTANCE), else None.
        """
        try:
            if self._tbl.count_rows() == 0:
                return None
            results = (
                self._tbl.search(embedding)
                .metric("cosine")
                .limit(1)
                .to_list()
            )
            if results and results[0].get("_distance", 1.0) < DEDUP_DISTANCE:
                hit = results[0]
                return hit["id"], float(hit.get("confidence") or 0.5)
        except Exception as exc:
            log.debug("Duplicate search failed (non-fatal): %s", exc)
        return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_fact(self, text: str,
                   confidence: float = 0.5,
                   source: str = "agent",
                   provenance: str = "",
                   time_frame: str = "permanent",
                   embedding: Optional[List[float]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   media_path: Optional[str] = None) -> str:
        """Persist a fact; skips insert if a semantic duplicate already exists.

        When a duplicate is found, the existing row is updated with the newer
        text, merged provenance, upgraded time_frame, and (if provided) media.

        Args:
            text:        The fact text.
            confidence:  HLD tag — certainty of this fact (0.2–1.0).
            source:      HLD tag — origin: "person", "agent", or stream name.
            provenance:  HLD tag — description of the inputs that led to this fact.
            time_frame:  HLD tag — when the fact is true (permanent, session,
                         today, recent, dated).
            embedding:   Pre-computed vector (generated internally if omitted).
            metadata:    Any extra unstructured fields.
            media_path:  HLD: "tgz file that may contain supporting media."
                         Path to a .tgz archive of images, audio, etc.
        """
        confidence = max(0.2, min(1.0, float(confidence)))

        vec = embedding if embedding is not None else self._embed(text)

        # Semantic deduplication — update the existing row with the newer
        # (presumably richer) text, upgraded time_frame, and merged provenance
        # so that improved ToM/profile facts replace stale approximations.
        if vec is not None:
            dup = self._find_duplicate(vec)
            if dup is not None:
                dup_id, dup_conf = dup
                new_conf = min(1.0, max(confidence, dup_conf + CONFIDENCE_BUMP))
                # Merge provenance from independent sources (deduped by
                # normalised source key so repeated replay entries don't
                # accumulate).
                existing_row = self._fetch_row(dup_id)
                merge_prov = provenance
                if existing_row is not None:
                    merge_prov = _merge_provenance(
                        existing_row.get("provenance", ""), provenance,
                    )
                # Upgrade time_frame if the incoming fact is more durable.
                merge_tf = time_frame
                if existing_row is not None:
                    old_tf = existing_row.get("time_frame", "permanent")
                    if _TF_RANK.get(old_tf, 0) > _TF_RANK.get(time_frame, 0):
                        merge_tf = old_tf
                self.update_fact(
                    dup_id,
                    text=text,
                    confidence=new_conf,
                    source=source,
                    provenance=merge_prov,
                    time_frame=merge_tf,
                    metadata=metadata,
                )
                log.debug("Dedup: updated '%s…' (matched %s, conf %.2f→%.2f)",
                          text[:50], dup_id[:8], dup_conf, new_conf)
                return dup_id

        fact_id = str(uuid.uuid4())
        self._tbl.add([{
            "id":            fact_id,
            "text":          text,
            "confidence":    confidence,
            "source":        source,
            "provenance":    provenance,
            "time_frame":    time_frame,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "metadata_json": json.dumps(metadata or {}),
            "media_path":    media_path or "",
            "vector":        vec if vec is not None else [0.0] * EMBED_DIM,
        }])
        self._fts_dirty = True
        log.debug("Stored fact %s (conf=%.2f src=%s): %s",
                  fact_id[:8], confidence, source, text[:60])
        return fact_id

    def _fetch_row(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single row by id using Arrow filtering (avoids loading pandas)."""
        try:
            table = self._tbl.to_arrow()
            mask = pc.equal(table.column("id"), fact_id)
            filtered = table.filter(mask)
            if len(filtered) == 0:
                return None
            cols = filtered.to_pydict()
            row = {k: v[0] for k, v in cols.items()}
            # Normalise vector: pyarrow may return it as a list-of-dicts or ndarray
            vec = row.get("vector")
            if isinstance(vec, np.ndarray):
                row["vector"] = vec.tolist()
            return row
        except Exception as exc:
            log.warning("_fetch_row failed for %s: %s", fact_id[:8], exc)
            return None

    def _upsert_row(self, row: Dict[str, Any]) -> None:
        """Atomically insert-or-replace a row by id.

        Uses LanceDB ``merge_insert`` so there is no delete-then-add window in
        which a crash (or a failed add) loses the row entirely (P1-c).  Falls
        back to delete+add only on an older LanceDB lacking merge_insert."""
        try:
            (self._tbl.merge_insert("id")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .execute([row]))
        except AttributeError:
            self._tbl.delete(f"id = '{row['id']}'")
            self._tbl.add([row])

    def _bump_confidence(self, fact_id: str, new_confidence: float) -> None:
        """Update only the confidence of an existing fact (atomic upsert)."""
        try:
            row = self._fetch_row(fact_id)
            if row is None:
                return
            row["confidence"] = new_confidence
            self._upsert_row(row)
            self._fts_dirty = True
        except Exception as exc:
            log.warning("Failed to bump confidence for %s: %s", fact_id[:8], exc)

    def delete_fact(self, fact_id: str) -> None:
        self._tbl.delete(f"id = '{fact_id}'")
        self._fts_dirty = True

    def update_fact(self, fact_id: str, text: Optional[str] = None,
                    confidence: Optional[float] = None,
                    source: Optional[str] = None,
                    provenance: Optional[str] = None,
                    time_frame: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    media_path: Optional[str] = None) -> bool:
        existing = self._fetch_row(fact_id)
        if existing is None:
            return False
        new_text = text if text is not None else existing["text"]
        new_confidence = max(0.2, min(1.0, float(confidence))) if confidence is not None \
                         else float(existing.get("confidence") or 0.5)
        new_source = source if source is not None else (existing.get("source") or "agent")
        new_provenance = provenance if provenance is not None else (existing.get("provenance") or "")
        new_time_frame = time_frame if time_frame is not None else (existing.get("time_frame") or "permanent")
        new_media = media_path if media_path is not None else (existing.get("media_path") or "")
        existing_meta = json.loads(existing.get("metadata_json", "{}"))
        if metadata:
            existing_meta.update(metadata)

        # Re-embed if text changed
        vec = existing.get("vector")
        if text is not None and text != existing["text"]:
            new_vec = self._embed(new_text)
            if new_vec is not None:
                vec = new_vec
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()

        # Atomic upsert (no delete-then-add window) and PRESERVE the original
        # creation timestamp — an update is not a re-creation (P1-c).
        self._upsert_row({
            "id":            fact_id,
            "text":          new_text,
            "confidence":    new_confidence,
            "source":        new_source,
            "provenance":    new_provenance,
            "time_frame":    new_time_frame,
            "timestamp":     existing.get("timestamp")
                             or datetime.now(timezone.utc).isoformat(),
            "metadata_json": json.dumps(existing_meta),
            "media_path":    new_media,
            "vector":        vec if vec is not None else [0.0] * EMBED_DIM,
        })
        self._fts_dirty = True
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search_text(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Full-text search. Falls back to substring match if FTS unavailable."""
        self._ensure_fts_index()
        try:
            rows = self._tbl.search(query, query_type="fts").limit(limit).to_list()
            return [self._to_fact(r) for r in rows]
        except Exception as exc:
            log.warning("FTS search failed, using substring fallback: %s", exc)
            df = self._tbl.to_pandas()
            mask = df["text"].str.contains(query, case=False, na=False)
            return [self._to_fact(r) for r in df[mask].tail(limit).to_dict("records")]

    def search_semantic(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Vector similarity search using the query embedding."""
        vec = self._embed(query)
        if vec is None:
            return self.search_text(query, limit)
        try:
            rows = (
                self._tbl.search(vec)
                .metric("cosine")
                .limit(limit)
                .to_list()
            )
            return [self._to_fact(r) for r in rows]
        except Exception as exc:
            log.warning("Vector search failed, falling back to FTS: %s", exc)
            return self.search_text(query, limit)

    # Keep search_context as an alias so IyyeBrain callers don't break
    search_context = search_text

    # LLM-generated streams (via the stream_codegen_goal/_sensor prompts)
    # call memory.search(); alias to search_semantic so they get the best
    # available results.
    search = search_semantic

    def get_recent_facts(self, limit: int = 100) -> List[Dict[str, Any]]:
        df = self._tbl.to_pandas().tail(limit)
        return [self._to_fact(r) for r in df.to_dict("records")]

    def count(self) -> int:
        return self._tbl.count_rows()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fts_index(self) -> None:
        if not self._fts_dirty:
            return
        try:
            self._tbl.create_fts_index("text", replace=True)
            self._fts_dirty = False
        except Exception as exc:
            log.warning("FTS index rebuild failed: %s", exc)

    def _to_fact(self, row: Dict[str, Any]) -> Dict[str, Any]:
        fact: Dict[str, Any] = {
            "id":         row["id"],
            "text":       row["text"],
            "confidence": float(row.get("confidence") or 0.5),
            "source":     row.get("source") or "agent",
            "provenance": row.get("provenance") or "",
            "time_frame": row.get("time_frame") or "permanent",
            "timestamp":  row["timestamp"],
            "metadata":   json.loads(row.get("metadata_json", "{}")),
        }
        mp = row.get("media_path") or ""
        if mp:
            fact["media_path"] = mp
        return fact

    def close(self) -> None:
        log.info("MemoryClient closed")
