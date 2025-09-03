"""Utility functions for the tool-calling-sft-mix project."""

import json
import orjson
from typing import Dict, Any, List

from src.quality_control import TOOLBENCH_FIELD_MAPPINGS


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


def adapt_toolbench_row_with_normalization(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt ToolBench dataset rows to standard format with field normalization."""
    from src.parsers import adapt_toolbench_row
    
    # First adapt using base adapter
    adapted = adapt_toolbench_row(row)
    if not adapted.get("valid"):
        return adapted
        
    # Then apply field normalization if needed
    try:
        target = json.loads(adapted["target_json"])
        tool_calls = target.get("tool_calls", [])
        
        # Normalize tool call fields
        for call in tool_calls:
            if "parameters" in call:
                params = call["parameters"]
                if isinstance(params, dict):
                    # Apply field normalization mappings
                    normalized = {}
                    for k, v in params.items():
                        # Check if this field has a normalized name
                        norm_key = TOOLBENCH_FIELD_MAPPINGS.get(k, k)
                        normalized[norm_key] = v
                    call["parameters"] = normalized
                    
        # Update target with normalized calls
        target["tool_calls"] = tool_calls
        adapted["target_json"] = json.dumps(target)
        adapted["meta_source"] = "toolbench_normalized"
        
    except Exception:
        # If normalization fails, return original adaptation
        pass
        
    return adapted


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


def add_difficulty(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Add difficulty classification based on n_calls and tool types."""
    try:
        # Parse target_json to get tool calls
        target = json.loads(ex.get("target_json", "{}"))
        tool_calls = target.get("tool_calls", [])
        
        # Count total calls and unique tools
        n_calls = len(tool_calls)
        unique_tools = len(set(call.get("name", "") for call in tool_calls))
        
        # Determine difficulty
        if n_calls == 0:
            diff = "no_call"
        elif n_calls == 1:
            diff = "simple"
        elif unique_tools > 1:
            diff = "parallel"  # Multiple different tools = parallel
        else:
            diff = "multiple"  # Multiple calls to same tool = multiple
            
        # Update example with correct counts and difficulty
        return {
            **ex,
            "n_calls": n_calls,
            "difficulty": diff,
            "valid": True if n_calls > 0 else ex.get("valid", False)
        }
        
    except Exception:
        # Keep existing values on error
        return {
            **ex,
            "difficulty": ex.get("difficulty", "simple")
        }