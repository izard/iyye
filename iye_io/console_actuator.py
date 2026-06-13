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
Simple console actuator – always available.
"""

import logging

from iyye_base import BaseActuator   # shared base class

log = logging.getLogger("Iyye.Actuators.Console")

class ConsoleActuator(BaseActuator):
    """
    Writes the payload to the Python logger (INFO level).  This acts as a
    minimal “actuator” that works even when no UI is running.
    """

    def _do_actuate(self, payload: str) -> None:
        log.info("[Console Actuator] %s", payload)

# The discovery mechanism will instantiate this class automatically.
