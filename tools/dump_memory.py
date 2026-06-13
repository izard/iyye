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
Debug script: print all facts from the Iyye long-term memory (LanceDB).
Run from the project root:  python dump_memory.py [db_path]
"""

import sys
import lancedb

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "./iyye_memory"
TABLE   = "facts"

db = lancedb.connect(DB_PATH)
if TABLE not in db.table_names():
    print(f"No '{TABLE}' table found in {DB_PATH}")
    sys.exit(0)

rows = db.open_table(TABLE).to_pandas()
total = len(rows)
print(f"=== Long-term memory: {total} fact(s) in {DB_PATH} ===\n")

for i, row in rows.iterrows():
    print(f"[{i+1}/{total}]")
    print(f"  id:         {row.get('id', '?')}")
    print(f"  timestamp:  {row.get('timestamp', '?')}")
    print(f"  confidence: {row.get('confidence', '?'):.2f}")
    print(f"  time_frame: {row.get('time_frame', '?')}")
    print(f"  source:     {row.get('source', '?')}")
    print(f"  provenance: {row.get('provenance', '?')}")
    mp = row.get('media_path', '')
    if mp:
        print(f"  media:      {mp}")
    print(f"  text:       {row.get('text', '?')}")
    print()
