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
"""IO sensors and actuators for Iyye.

This package is loaded both as a normal Python package (for direct imports
like ``from iyye_io.memory_mcp_client import MemoryClient``) and scanned by
``IyyeBrain._discover_io()`` which dynamically loads every .py file and
instantiates BaseSensorQueue / BaseActuator subclasses.

The __init__ intentionally does NOT re-export sensor/actuator classes so that
the dynamic loader remains the single registration path — no risk of double-
instantiation.
"""
