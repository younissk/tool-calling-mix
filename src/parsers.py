"""Parsing and adaptation logic for different dataset formats."""

import ast
import re
import orjson
from typing import Dict, Any, Optional

from src.utils import json_dumps, make_empty_row, make_target


def safe_eval_pythonish(s: Any) -> Optional[Any]:
    """Parse Python-ish dict/list strings (single quotes) safely."""
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return None
    
    try:
        return ast.literal_eval(s)
    except Exception:
        # Last ditch: try to coerce single→double quotes when it looks like a dict
        if s.strip().startswith("{") or s.strip().startswith("["):
            try:
                j = s.replace("'", '"')
                return orjson.loads(j.encode('utf-8'))
            except Exception:
                return None
        return None


def to_param_schema(parameters: Any) -> Any:
    """
    Normalize parameters which may be:
      - dict like {'start': {...}, 'end': {...}}
      - list like [{'name':'start', ...}, ...]
      - or something else (return as-is)
    """
    if isinstance(parameters, dict):
        # assume already object-ish
        return {"type": "object", "properties": parameters}
    
    if isinstance(parameters, list):
        props = {}
        for p in parameters:
            if isinstance(p, dict) and "name" in p:
                nm = p["name"]
                pd = {k: v for k, v in p.items() if k != "name"}
                props[nm] = pd
        if props:
            return {"type": "object", "properties": props}
    
    return parameters


_CALL_RE = re.compile(r'^\s*([A-Za-z0-9_.]+)\s*\((.*)\)\s*$')


def parse_call_string(s: str) -> tuple[Optional[str], Dict[str, Any]]:
    """
    Parse 'api.func(a=1, b="x", arr=["y"])' → ("api.func", {a:1, b:"x", arr:["y"]})
    """
    if not isinstance(s, str):
        return None, {}
    
    m = _CALL_RE.match(s.strip())
    if not m:
        return None, {}
    
    func = m.group(1)
    args_str = m.group(2).strip()
    
    # make JSON-ish
    jsonish = re.sub(r'(\w+)\s*=', r'"\1":', args_str)
    jsonish = jsonish.replace("None", "null").replace("True", "true").replace("False", "false")
    
    try:
        args = orjson.loads(("{" + jsonish + "}").encode('utf-8'))
    except Exception:
        # try Python eval for tricky literals
        try:
            tmp = ast.literal_eval("{" + args_str + "}")
            args = tmp if isinstance(tmp, dict) else {"_raw": args_str}
        except Exception:
            args = {"_raw": args_str}
    
    return func, args


def adapt_xlam60k(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt xLAM 60k dataset rows to standard format."""
    out = make_empty_row()
    tools = row.get("tools") or row.get("functions") or []
    messages = row.get("messages") or []

    tc = None
    for m in messages[::-1]:
        if isinstance(m, dict) and m.get("role") in ("assistant", "tool"):
            if m.get("tool_calls"):
                tc = m["tool_calls"]
                break
            if m.get("function_call"):
                f = m["function_call"]
                tc = [{"name": f.get("name"), "arguments": f.get("arguments")}]
                break
    
    if tc is None:
        if "tool_calls" in row:
            tc = row["tool_calls"]
        elif "function_call" in row:
            f = row["function_call"]
            tc = [{"name": f.get("name"), "arguments": f.get("arguments")}]

    if not tc:
        return out

    target = make_target(tc)
    n = len(target.get("tool_calls", []))
    if n == 0:
        return out

    if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
        return out

    out["tools_json"] = json_dumps(tools)
    out["messages_json"] = json_dumps(messages)
    out["target_json"] = json_dumps(target)
    out["meta_source"] = "xlam60k"
    out["n_calls"] = n
    out["valid"] = True
    return out


def adapt_openfunctions_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt OpenFunctions dataset rows to standard format."""
    out = make_empty_row()

    # Handle Instruction/Functions/Output format
    if isinstance(row.get("Instruction"), str) and isinstance(row.get("Functions"), list) and row.get("Output"):
        user_text = row["Instruction"]
        tool_specs = []
        for f in row["Functions"]:
            parsed = safe_eval_pythonish(f)
            if isinstance(parsed, dict):
                name = parsed.get("api_name") or parsed.get("api_call") or parsed.get("name") or "function_call"
                tool_specs.append({
                    "name": name,
                    "description": parsed.get("description", ""),
                    "parameters": to_param_schema(parsed.get("parameters", {}))
                })

        tool_calls = []
        outs = row.get("Output") or []
        if isinstance(outs, str):
            outs = [outs]
        for s in outs:
            func, args = parse_call_string(s)
            if func:
                tool_calls.append({"name": func, "arguments": args})

        if not tool_calls:
            return out

        messages = [{"role": "user", "content": user_text}]
        target = make_target(tool_calls)
        n = len(target.get("tool_calls", []))
        if n == 0:
            return out

        out["tools_json"] = json_dumps(tool_specs)
        out["messages_json"] = json_dumps(messages)
        out["target_json"] = json_dumps(target)
        out["meta_source"] = "openfunctions_v1"
        out["n_calls"] = n
        out["valid"] = True
        return out

    # Handle question/function/model_answer format
    if isinstance(row.get("question"), str) and isinstance(row.get("function"), dict) and "model_answer" in row:
        user_text = row["question"]
        fn = row["function"] or {}
        tool_name = fn.get("api_call") or fn.get("api_name") or fn.get("name") or "function_call"
        tools = [{
            "name": tool_name,
            "description": fn.get("description", ""),
            "parameters": to_param_schema(fn.get("parameters", {}))
        }]
        func_used, args = parse_call_string(row["model_answer"])
        call_name = func_used or tool_name
        target = make_target([{"name": call_name, "arguments": args}])
        n = len(target.get("tool_calls", []))
        if n == 0:
            return out
        messages = [{"role": "user", "content": user_text}]
        out["tools_json"] = json_dumps(tools)
        out["messages_json"] = json_dumps(messages)
        out["target_json"] = json_dumps(target)
        out["meta_source"] = "openfunctions_v1"
        out["n_calls"] = n
        out["valid"] = True
        return out

    # Handle generic format
    tools = row.get("functions") or row.get("tools") or row.get("tool_desc") or []
    user_text = row.get("instruction") or row.get("user") or row.get("prompt") or row.get("question")
    if user_text:
        messages = [{"role": "user", "content": str(user_text)}]
        raw = row.get("output") or row.get("response") or row.get("assistant") or row.get("tool_calls")
        tool_calls = None
        if isinstance(raw, list):
            tool_calls = raw
        elif isinstance(raw, dict):
            tool_calls = raw.get("tool_calls") or raw
        elif isinstance(raw, str):
            try:
                parsed = orjson.loads(raw.encode('utf-8'))
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = parsed["tool_calls"]
                elif isinstance(parsed, (list, dict)):
                    tool_calls = parsed
            except Exception:
                pass
        if tool_calls:
            target = make_target(tool_calls)
            n = len(target.get("tool_calls", []))
            if n > 0:
                out["tools_json"] = json_dumps(tools)
                out["messages_json"] = json_dumps(messages)
                out["target_json"] = json_dumps(target)
                out["meta_source"] = "openfunctions_v1"
                out["n_calls"] = n
                out["valid"] = True
                return out

    return out
