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
Camera Sensor — captures webcam snapshots via ffmpeg (macOS AVFoundation)
and describes them using the multimodal Gemma-4 vision LLM.

HLD: "camera and multimodal LLM that translates visual snapshots to text
      2-3 times per minute."

Runs as a background thread, capturing every `capture_interval` seconds.
Each description is pushed into the sensor queue so the brain can read it
the same way as any other sensor input.

Requirements:
  - ffmpeg in PATH (comes with miniforge/conda)
  - Vision LLM server running on port 8081 (start with tools/llm-gemma4-vision.sh)

Environment variables:
  CAMERA_DEVICE           AVFoundation device index (default: 0 = FaceTime HD Camera)
  CAMERA_INTERVAL         Capture interval in seconds (default: 25)
  LLM_VISION_API_BASE     Vision server URL (default: http://127.0.0.1:8081/v1)
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from iyye_base import PROJECT_ROOT, BaseSensorQueue

log = logging.getLogger("Iyye.Sensors.Camera")

_FFMPEG = "ffmpeg"
_DEFAULT_DEVICE   = int(os.environ.get("CAMERA_DEVICE", "0"))
_DEFAULT_INTERVAL = float(os.environ.get("CAMERA_INTERVAL", "25"))


def _capture_frame(device: int = _DEFAULT_DEVICE) -> Optional[bytes]:
    """
    Capture a single JPEG frame from the webcam using ffmpeg AVFoundation.
    Returns raw JPEG bytes or None on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                _FFMPEG, "-y",
                "-f", "avfoundation",
                "-framerate", "1",
                "-video_size", "640x480",
                "-i", f"{device}",
                "-vframes", "1",
                "-q:v", "5",          # JPEG quality 1-31, lower = better
                tmp_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if result.returncode != 0:
            log.warning("Camera: ffmpeg returned %d", result.returncode)
            return None
        with open(tmp_path, "rb") as fh:
            data = fh.read()
        return data if data else None
    except subprocess.TimeoutExpired:
        log.warning("Camera: ffmpeg capture timed out")
        return None
    except Exception as exc:
        log.warning("Camera: capture failed: %s", exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class CameraSensor(BaseSensorQueue):
    """
    Periodically captures webcam frames, describes them via VisionClient,
    and pushes text descriptions into the sensor queue.
    """

    def __init__(
        self,
        name: str = "camera_sensor",
        maxlen: int = 100,
        capture_interval: float = _DEFAULT_INTERVAL,
        device: int = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(name=name, maxlen=maxlen)
        self._interval = capture_interval
        self._device   = device
        self._thread: Optional[threading.Thread] = None
        self._running  = False
        self._vision: Optional[object] = None   # VisionClient, lazy-loaded

    # ------------------------------------------------------------------
    # BaseSensorQueue hooks
    # ------------------------------------------------------------------

    def start_collection(self) -> None:
        """Start the background capture thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="camera_capture", daemon=True
        )
        self._thread.start()
        log.info("CameraSensor started (device=%d, interval=%.0fs)", self._device, self._interval)

    def stop_collection(self) -> None:
        self._running = False
        log.info("CameraSensor stopped")

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self._capture_and_push()
            except Exception as exc:
                log.warning("CameraSensor loop error: %s", exc)
            time.sleep(self._interval)

    def _capture_and_push(self) -> None:
        jpeg = _capture_frame(self._device)
        if jpeg is None:
            return

        vision = self._get_vision()
        if vision is None:
            log.debug("CameraSensor: vision LLM unavailable, skipping")
            return

        try:
            description = vision.describe_image_bytes(
                jpeg,
                mime="image/jpeg",
                prompt="Describe what you see. Focus on people, objects, activities, and any visible text.",
            )
        except Exception as exc:
            log.warning("CameraSensor: vision LLM failed: %s", exc)
            return

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source":    "camera",
            "text":      description,
        }
        self.push(payload)
        log.info("CameraSensor: captured description (%d chars)", len(description))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_vision(self):
        if self._vision is None:
            try:
                from llm_vision_client import VisionClient
                api_base = self._vision_api_base()
                client = VisionClient(api_base=api_base) if api_base else VisionClient()
                if client.is_healthy():
                    self._vision = client
                    log.info("CameraSensor: vision LLM connected at %s", client.api_base)
                else:
                    log.debug("CameraSensor: vision server not ready yet")
            except Exception as exc:
                log.debug("CameraSensor: VisionClient unavailable: %s", exc)
        return self._vision

    @staticmethod
    def _vision_api_base() -> Optional[str]:
        """Read llm-registry.json to find the default vision model port."""
        registry_path = PROJECT_ROOT / "tools" / "llm-registry.json"
        try:
            import json
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            for m in registry:
                if m.get("vision") and "vision" in m.get("default_for", []):
                    port = m["port"]
                    host = os.environ.get("LLM_HOST", "127.0.0.1")
                    return f"http://{host}:{port}/v1"
        except Exception:
            pass
        return None  # VisionClient will use its own default (env var or 8081)
