"""Utility functions for the tool-calling-sft-mix project."""

import json
import orjson
from typing import Dict, Any, List


def json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string using orjson with fallback to json."""
    try:
        return orjson.dumps(obj).decode('utf-8')
    except Exception:
        return json.dumps(obj, ensure_ascii=False)


def make_empty_row() -> Dict[str, Any]:
    """Create an empty row template for dataset adaptation."""
    return {
        "tools_json": "[]",
        "messages_json": "[]",
        "target_json": json_dumps({"tool_calls": []}),
        "meta_source": "",
        "n_calls": 0,
        "difficulty": "simple",
        "valid": False,
    }


def make_target(tool_calls: Any) -> Dict[str, Any]:
    """Convert tool calls to canonical format."""
    canon = []
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    if not isinstance(tool_calls, list):
        return {"tool_calls": []}
    
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        args = tc.get("arguments") or (tc.get("function") or {}).get("arguments") or {}
        
        if isinstance(args, str):
            try:
                args = orjson.loads(args)
            except Exception:
                args = {"_raw": args}
        
        canon.append({
            "name": str(name) if name is not None else "",
            "arguments": args
        })
    
    return {"tool_calls": canon}


def read_json_file(path: str) -> List[Dict[str, Any]]:
    """Read JSON or JSONL file and return list of dictionaries."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(2048)
        
        if head.lstrip().startswith("["):
            # JSON array format
            with open(path, "r", encoding="utf-8") as f:
                data = orjson.loads(f.read().encode('utf-8'))
            if isinstance(data, list):
                rows = data
        else:
            # JSONL format
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(orjson.loads(line.encode('utf-8')))
                    except Exception:
                        pass
    except Exception as e:
        print(f"[warn] cannot read {path}: {e}")
    
    return rows


def add_difficulty(ex: Dict[str, Any]) -> Dict[str, str]:
    """Add difficulty classification based on n_calls and user phrasing."""
    msg = ex["messages_json"].lower()
    if ex["n_calls"] <= 1:
        diff = "simple"
    elif any(k in msg for k in ("in parallel", "simultaneously", "at the same time", "parallel")):
        diff = "parallel"
    else:
        diff = "multiple"
    return {"difficulty": diff}
