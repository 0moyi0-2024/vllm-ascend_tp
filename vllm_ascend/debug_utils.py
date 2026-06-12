# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import os
from collections import defaultdict
from typing import Any

from vllm.logger import logger


_DEBUG_COUNTERS: defaultdict[str, int] = defaultdict(int)


def env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in ("", "0", "false", "off", "no")


def gemma4_graph_debug_enabled() -> bool:
    return env_flag_enabled("VLLM_ASCEND_GEMMA4_GRAPH_DEBUG")


def gemma4_graph_debug_limit() -> int:
    value = os.getenv("VLLM_ASCEND_GEMMA4_GRAPH_DEBUG_LIMIT", "600")
    try:
        return max(int(value), 0)
    except ValueError:
        return 600


def log_gemma4_graph_debug(key: str, message: str, *args: Any) -> None:
    if not gemma4_graph_debug_enabled():
        return

    limit = gemma4_graph_debug_limit()
    if limit and _DEBUG_COUNTERS[key] >= limit:
        return

    _DEBUG_COUNTERS[key] += 1
    logger.info("[gemma4-graph-debug] " + message, *args)
