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
# -*- coding: utf-8 -*-

"""
Hardware Sensor - Monitors CPU, GPU, Memory, and Disk usage.

HLD: "Platform CPU, GPU, Memory, disk usage sensor."
"""

import threading
import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from iyye_base import PROJECT_ROOT, BaseSensorQueue

log = logging.getLogger("Iyye.Sensors.Hardware")


class HardwareSensor(BaseSensorQueue):
    """
    Monitors system hardware metrics: CPU, Memory, Disk, and optionally GPU.
    HLD: "Platform CPU, GPU, Memory, disk usage sensor."
    """
    
    def __init__(self, name: str = "hardware_sensor",
                 poll_interval: float = 5.0,
                 maxlen: int = 10_000):
        super().__init__(name=name, maxlen=maxlen)
            
        self.poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread | None = None
        
    def start_collection(self) -> None:
        """Start background hardware monitoring."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._collect_loop, name="hw_sensor", daemon=True
            )
            self._thread.start()
            log.info("Hardware sensor started (interval=%.1fs)", self.poll_interval)
    
    def _collect_loop(self) -> None:
        """Background loop that collects hardware metrics."""
        while self._running:
            try:
                snapshot = self._collect_snapshot()
                self.push(snapshot)
            except Exception as e:
                log.error("Error collecting hardware data: %s", e)
            time.sleep(self.poll_interval)
    
    def _collect_snapshot(self) -> Dict[str, Any]:
        """Collect a single hardware snapshot."""
        vm = psutil.virtual_memory()
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": vm.percent,
            "memory_available_gb": vm.available / (1024**3),
            "memory_total_gb": vm.total / (1024**3),
            "disk_percent": psutil.disk_usage('/').percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
        }

        # Try to get GPU metrics if available
        gpu_info = self._get_gpu_info()
        if gpu_info:
            snapshot["gpu"] = gpu_info

        # LLM model inventory from llm-active.json (written by LlmManagementStream)
        snapshot["llm_models"] = self._read_llm_active(vm.total / (1024**3))

        return snapshot

    def _read_llm_active(self, ram_total_gb: float) -> Dict[str, Any]:
        """Read tools/llm-active.json for LLM model status."""
        import json
        active_path = str(PROJECT_ROOT / "tools" / "llm-active.json")
        try:
            with open(active_path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.debug("HardwareSensor: could not read llm-active.json: %s", exc)
        # File not written yet — return skeleton
        return {
            "updated": None,
            "ram_total_gb": round(ram_total_gb, 1),
            "ram_available_gb": None,
            "loaded_gb": 0,
            "headroom_gb": None,
            "active_count": 0,
            "models": [],
        }
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Try to get GPU information (NVIDIA only)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            pynvml.nvmlShutdown()
            
            return {
                "gpu_percent": gpu_util.gpu,
                "gpu_memory_percent": (mem_info.used / mem_info.total) * 100,
            }
        except Exception:
            # GPU monitoring not available
            return None
    
    def stop_collection(self) -> None:
        """Stop hardware monitoring."""
        self._running = False
        log.info("Hardware sensor stopped")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current hardware stats without storing."""
        return self._collect_snapshot()



