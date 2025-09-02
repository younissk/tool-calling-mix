"""Data loading functions for different datasets."""

import os
import orjson
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from typing import Optional

from src.config import (
    USE_XLAM, CAP_XLAM, CAP_OPENFUNCTIONS, CAP_DOLLY_NO_CALL, 
    CAP_WIKITEXT_NO_CALL, RANDOM_SEED
)
from src.parsers import adapt_xlam60k, adapt_openfunctions_row
from src.utils import make_empty_row, read_json_file, add_difficulty


def load_xlam60k(cap: int = CAP_XLAM) -> Optional[Dataset]:
    """Load and adapt xLAM 60k dataset."""
    if not USE_XLAM:
        return None
    
    try:
        ds = load_dataset("minpeter/xlam-function-calling-60k-parsed", split="train")
    except Exception as e:
        print("[skip] xLAM load failed:", e)
        return None

    def _map(row):
        return adapt_xlam60k(row)
    
    mapped = ds.map(_map, remove_columns=ds.column_names, desc="Adapting xLAM")  # type: ignore
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    
    if cap and len(mapped) > cap:
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))  # type: ignore
    
    print(f"[ok] xLAM: {len(mapped)}")
    return mapped  # type: ignore


def load_openfunctions_local(files: list[str], cap: int = CAP_OPENFUNCTIONS) -> Optional[Dataset]:
    """Load and adapt OpenFunctions dataset from local files."""
    # 1) Read all rows into Python dicts
    all_rows = []
    for fp in files:
        if not os.path.exists(fp):
            print(f"[skip] missing: {fp}")
            continue
        rows = read_json_file(fp)
        if not rows:
            print(f"[warn] empty/parse-failed: {fp}")
            continue
        all_rows.extend(rows)
        print(f"[ok] loaded {len(rows)} from {os.path.basename(fp)}")
    
    if not all_rows:
        return None

    # 2) Create a dataset with a single uniform column to avoid Arrow mixed types
    ds = Dataset.from_dict({"raw_json": [orjson.dumps(r).decode('utf-8') for r in all_rows]})

    # 3) Map: parse, adapt, drop raw_json
    def _map(row):
        try:
            obj = orjson.loads(row["raw_json"].encode('utf-8'))
        except Exception:
            return make_empty_row()
        return adapt_openfunctions_row(obj)

    mapped = ds.map(_map, remove_columns=["raw_json"], desc="Adapting OpenFunctions v1")
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    mapped = mapped.map(add_difficulty)

    if cap and len(mapped) > cap:
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))
    
    print(f"[ok] OpenFunctions v1 (local): {len(mapped)}")
    return mapped


def load_dolly_nocall(cap: int = CAP_DOLLY_NO_CALL) -> Optional[Dataset]:
    """
    Convert Dolly-15k instruction-following into 'no-call' targets.
    This teaches the model that some instructions should yield NO tool calls.
    """
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print("[skip] Dolly load failed:", e)
        return None

    def _map(row):
        instr = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        user = (instr + ("\n" + ctx if ctx else "")).strip()
        
        if not user:
            return {
                "tools_json": "[]", 
                "messages_json": "[]",
                "target_json": orjson.dumps({"tool_calls": []}).decode('utf-8'),
                "meta_source": "instr_nocall_dolly",
                "n_calls": 0, 
                "difficulty": "simple", 
                "valid": False,
            }
        
        messages = [{"role": "user", "content": user}]
        return {
            "tools_json": "[]",
            "messages_json": orjson.dumps(messages).decode('utf-8'),
            "target_json": orjson.dumps({"tool_calls": []}).decode('utf-8'),
            "meta_source": "instr_nocall_dolly",
            "n_calls": 0, 
            "difficulty": "simple", 
            "valid": True,
        }

    mapped = ds.map(_map, remove_columns=ds.column_names, desc="Adapting Dolly->no-call")  # type: ignore
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    
    if cap and len(mapped) > cap:
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))  # type: ignore
    
    print(f"[ok] Dolly no-call: {len(mapped)}")
    return mapped  # type: ignore


def load_wikitext_nocall(cap: int = CAP_WIKITEXT_NO_CALL) -> Optional[Dataset]:
    """Load WikiText and convert to no-call format for catastrophic forgetting prevention."""
    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print("[skip] WikiText load failed:", e)
        return None

    def _map(row):
        text = (row.get("text") or "").strip()
        if not text:
            return {
                "tools_json": "[]",
                "messages_json": "[]",
                "target_json": orjson.dumps({"tool_calls": []}).decode('utf-8'),
                "meta_source": "wikitext_nocall",
                "n_calls": 0,
                "difficulty": "simple",
                "valid": False
            }
        
        # Use a short excerpt as a 'user' message to regularize language understanding
        user = text[:512]
        msgs = [{"role": "user", "content": user}]
        return {
            "tools_json": "[]",
            "messages_json": orjson.dumps(msgs).decode('utf-8'),
            "target_json": orjson.dumps({"tool_calls": []}).decode('utf-8'),
            "meta_source": "wikitext_nocall",
            "n_calls": 0,
            "difficulty": "simple",
            "valid": True
        }

    mapped = ds.map(_map, remove_columns=ds.column_names, desc="Adapting WikiText->no-call")  # type: ignore
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    
    if cap and len(mapped) > cap:
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))  # type: ignore
    
    print(f"[ok] WikiText no-call: {len(mapped)}")
    return mapped  # type: ignore


def ensure_hub_compatible_schema(dataset: Dataset) -> Dataset:
    """
    Ensure the dataset has a schema compatible with Hugging Face Hub.
    This fixes the ArrowNotImplementedError that occurs when Hub tries to convert
    Arrow files to Parquet for the dataset viewer.
    """
    # Remove any internal columns that start with underscore
    bad_columns = [c for c in dataset.column_names if c.startswith("_")]
    if bad_columns:
        print(f"Removing internal columns: {bad_columns}")
        dataset = dataset.remove_columns(bad_columns)
    
    # Define Parquet-friendly features
    features = Features({
        "tools_json": Value("string"),
        "messages_json": Value("string"), 
        "target_json": Value("string"),
        "meta_source": Value("string"),
        "n_calls": Value("int32"),  # Use int32 instead of int64 for Parquet compatibility
        "difficulty": Value("string"),
        "valid": Value("bool"),
    })
    
    # Cast to ensure Parquet-friendly types
    print("Ensuring Parquet-compatible schema...")
    dataset = dataset.cast(features)
    
    # Clear any active formatting/transforms
    dataset.reset_format()
    
    return dataset


def create_mixed_dataset() -> Dataset:
    """Create the final mixed dataset by combining all available datasets."""
    from src.config import OPENFUNCTIONS_FILES
    
    parts = []

    # Load xLAM dataset
    xlam_ds = load_xlam60k()
    if xlam_ds is not None and len(xlam_ds) > 0:
        parts.append(xlam_ds)

    # Load OpenFunctions dataset
    openfunc_ds = load_openfunctions_local(OPENFUNCTIONS_FILES)
    if openfunc_ds is not None and len(openfunc_ds) > 0:
        parts.append(openfunc_ds)

    # Load Dolly no-call dataset
    dolly_nocall = load_dolly_nocall()
    if dolly_nocall is not None and len(dolly_nocall) > 0:
        parts.append(dolly_nocall)

    # Load WikiText no-call dataset
    wtx = load_wikitext_nocall()
    if wtx is not None and len(wtx) > 0:
        parts.append(wtx)

    if not parts:
        raise RuntimeError("No data loaded. Check paths and try again.")

    mix = concatenate_datasets(parts).shuffle(seed=RANDOM_SEED)
    print("TOTAL adapted:", len(mix))
    
    # Ensure Hub-compatible schema
    mix = ensure_hub_compatible_schema(mix)
    
    return mix
