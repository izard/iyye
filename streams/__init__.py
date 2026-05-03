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
# streams/__init__.py
"""Processing streams for Iyye consciousness model.

This module provides the special subconscious streams defined in the HLD:
- AttentionStream: Selects which stream becomes conscious
- AlignmentStream: Scores streams against high-level goals
- StreamFactory: Creates new streams for high-alignment opportunities
- SelfReflectionStream: Monitors system state (can become conscious)
- AdenosineStream: Manages tiredness/energy metric
- LlmManagementStream: Starts/stops/monitors local LLMs (HLD stream #6)
- UserChatStream: Processes user messages via LLM (created by StreamFactory)
"""

from .attention_stream import AttentionStream
from .alignment_stream import AlignmentStream
from .stream_factory import StreamFactory
from .self_reflection_stream import SelfReflectionStream
from .adenosine_stream import AdenosineStream
from .llm_management_stream import LlmManagementStream
from .user_chat_stream import UserChatStream

__all__ = [
    'AttentionStream',
    'AlignmentStream',
    'StreamFactory',
    'SelfReflectionStream',
    'AdenosineStream',
    'LlmManagementStream',
    'UserChatStream',
]

