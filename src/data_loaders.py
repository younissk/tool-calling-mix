"""Data loading functions for different datasets."""

import os
import time
import orjson
from datasets import load_dataset, Dataset, concatenate_datasets, Features, Value
from typing import Optional
from tqdm import tqdm

from src.config import (
    USE_XLAM, USE_TOOLBENCH, USE_SYNTHETIC_PARALLEL, CAP_XLAM, CAP_OPENFUNCTIONS, CAP_TOOLBENCH,
    CAP_DOLLY_NO_CALL, CAP_WIKITEXT_NO_CALL, CAP_SYNTHETIC_PARALLEL, RANDOM_SEED
)
from src.parsers import adapt_xlam60k, adapt_openfunctions_row, adapt_toolbench_row
from src.utils import make_empty_row, read_json_file, add_difficulty


def load_xlam60k(cap: int = CAP_XLAM) -> Optional[Dataset]:
    """Load and adapt xLAM 60k dataset."""
    if not USE_XLAM:
        return None

    try:
        ds = load_dataset(
            "minpeter/xlam-function-calling-60k-parsed", split="train")
    except Exception as e:
        print("[skip] xLAM load failed:", e)
        return None

    def _map(row):
        return adapt_xlam60k(row)

    mapped = ds.map(_map, remove_columns=ds.column_names,  # type: ignore
                    desc="Adapting xLAM")  # type: ignore
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))

    if cap and len(mapped) > cap:
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(  # type: ignore
            range(cap))  # type: ignore

    print(f"[ok] xLAM: {len(mapped)}")
    return mapped  # type: ignore


def load_openfunctions_local(files: list[str], cap: int = CAP_OPENFUNCTIONS) -> Optional[Dataset]:
    """Load and adapt OpenFunctions dataset from local files."""
    print(f"[start] Loading OpenFunctions from {len(files)} files...")
    start_time = time.time()

    # 1) Read all rows into Python dicts
    all_rows = []
    print("[step 1/4] Reading JSON files...")

    for fp in tqdm(files, desc="Reading OpenFunctions files", unit="file"):
        if not os.path.exists(fp):
            print(f"[skip] missing: {fp}")
            continue

        file_start = time.time()
        rows = read_json_file(fp)
        file_time = time.time() - file_start

        if not rows:
            print(f"[warn] empty/parse-failed: {fp}")
            continue

        all_rows.extend(rows)
        file_size = os.path.getsize(fp) / (1024 * 1024)  # MB
        print(
            f"[ok] loaded {len(rows)} rows from {os.path.basename(fp)} ({file_size:.1f}MB, {file_time:.2f}s)")

    if not all_rows:
        print("[error] No OpenFunctions data loaded")
        return None

    print(f"[step 2/4] Creating dataset from {len(all_rows)} rows...")
    dataset_start = time.time()

    # 2) Create a dataset with a single uniform column to avoid Arrow mixed types
    ds = Dataset.from_dict(
        {"raw_json": [orjson.dumps(r).decode('utf-8') for r in all_rows]})
    dataset_time = time.time() - dataset_start
    print(f"[ok] Dataset created in {dataset_time:.2f}s")

    # 3) Map: parse, adapt, drop raw_json
    print("[step 3/4] Adapting OpenFunctions rows...")
    adapt_start = time.time()

    def _map(row):
        try:
            obj = orjson.loads(row["raw_json"].encode('utf-8'))
        except Exception:
            return make_empty_row()
        return adapt_openfunctions_row(obj)

    mapped = ds.map(_map, remove_columns=["raw_json"], desc="Adapting OpenFunctions v1")
    adapt_time = time.time() - adapt_start
    print(f"[ok] Adaptation completed in {adapt_time:.2f}s")

    print("[step 4/4] Filtering and finalizing...")
    filter_start = time.time()

    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    mapped = mapped.map(add_difficulty)

    if cap and len(mapped) > cap:
        print(f"[cap] Limiting to {cap} examples (from {len(mapped)})")
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))

    filter_time = time.time() - filter_start
    total_time = time.time() - start_time

    print(
        f"[ok] OpenFunctions v1 (local): {len(mapped)} examples loaded in {total_time:.2f}s total")
    print(f"    - Dataset creation: {dataset_time:.2f}s")
    print(f"    - Adaptation: {adapt_time:.2f}s")
    print(f"    - Filtering: {filter_time:.2f}s")

    return mapped


def load_toolbench_local(files: list[str], cap: int = CAP_TOOLBENCH) -> Optional[Dataset]:
    """Load and adapt ToolBench dataset from local files."""
    if not USE_TOOLBENCH:
        print("[skip] ToolBench disabled in config")
        return None

    print(f"[start] Loading ToolBench from {len(files)} files...")
    start_time = time.time()

    # 1) Process files one at a time to avoid memory issues
    all_datasets = []
    print("[step 1/4] Reading and processing JSON files...")

    for fp in tqdm(files, desc="Reading ToolBench files", unit="file"):
        if not os.path.exists(fp):
            print(f"[skip] missing: {fp}")
            continue

        file_start = time.time()
        rows = read_json_file(fp)
        file_time = time.time() - file_start

        if not rows:
            print(f"[warn] empty/parse-failed: {fp}")
            continue

        file_size = os.path.getsize(fp) / (1024 * 1024)  # MB
        print(f"[ok] loaded {len(rows)} rows from {os.path.basename(fp)} ({file_size:.1f}MB, {file_time:.2f}s)")
        
        # Process this file in chunks to avoid memory issues
        print(f"[process] Processing {len(rows)} rows from {os.path.basename(fp)} in chunks...")
        
        chunk_size = 10000  # Process 10k rows at a time
        file_datasets = []
        
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i + chunk_size]
            print(f"[chunk] Processing chunk {i//chunk_size + 1}/{(len(rows) + chunk_size - 1)//chunk_size} ({len(chunk)} rows)")
            
            # Convert chunk to JSON strings
            json_chunk = [orjson.dumps(r).decode('utf-8') for r in chunk]
            
            # Create dataset for this chunk
            chunk_ds = Dataset.from_dict({"raw_json": json_chunk})
            
            # Adapt this chunk's data
            def _map(row):
                try:
                    obj = orjson.loads(row["raw_json"].encode('utf-8'))
                    return adapt_toolbench_row(obj)
                except Exception:
                    return make_empty_row()

            adapted_chunk = chunk_ds.map(_map, remove_columns=["raw_json"], desc=f"Adapting chunk {i//chunk_size + 1}")
            file_datasets.append(adapted_chunk)
            
            # Clear memory
            del chunk, json_chunk, chunk_ds
        
        # Combine chunks from this file
        if file_datasets:
            from datasets import concatenate_datasets
            file_ds = concatenate_datasets(file_datasets)
            all_datasets.append(file_ds)
            print(f"[ok] Processed {len(file_ds)} examples from {os.path.basename(fp)}")
        
        # Clear memory
        del rows, file_datasets

    if not all_datasets:
        print("[error] No ToolBench data loaded")
        return None

    print(f"[step 2/4] Combining {len(all_datasets)} processed datasets...")
    dataset_start = time.time()
    
    # Combine all datasets
    from datasets import concatenate_datasets
    ds = concatenate_datasets(all_datasets)
    dataset_time = time.time() - dataset_start
    print(f"[ok] Combined dataset created in {dataset_time:.2f}s")

    # 3) Filtering and finalizing (adaptation already done per file)
    print("[step 3/4] Filtering and finalizing...")
    filter_start = time.time()
    
    # The dataset is already adapted, just need to filter
    mapped = ds

    # Debug: count valid vs invalid before filtering
    try:
        # Convert to list to avoid indexing issues with Arrow dataset
        sample_batch = mapped.select(range(min(1000, len(mapped))))
        valid_sample = sum(
            1 for ex in sample_batch if getattr(ex, "valid", False))
        print(
            f"[debug] Sample validation rate: {valid_sample}/{len(sample_batch)} ({valid_sample/len(sample_batch)*100:.1f}%)")
    except Exception as e:
        print(f"[debug] Could not sample validation rate: {e}")

    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    mapped = mapped.map(add_difficulty)

    if cap and len(mapped) > cap:
        print(f"[cap] Limiting to {cap} examples (from {len(mapped)})")
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))

    filter_time = time.time() - filter_start
    total_time = time.time() - start_time

    print(
        f"[ok] ToolBench: {len(mapped)} examples loaded in {total_time:.2f}s total")
    print(f"    - Dataset creation: {dataset_time:.2f}s")
    print(f"    - Filtering: {filter_time:.2f}s")

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


def load_synthetic_parallel_data(cap: int = CAP_SYNTHETIC_PARALLEL) -> Optional[Dataset]:
    """Load or generate synthetic parallel tool call dataset."""
    if not USE_SYNTHETIC_PARALLEL:
        print("[skip] Synthetic parallel data disabled in config")
        return None

    print("[start] Loading synthetic parallel tool call data...")
    start_time = time.time()

    # Check if cached data exists
    cache_file = "data/synthetic_parallel_tool_calls.json"
    if os.path.exists(cache_file):
        print(f"[cache] Loading cached synthetic data from {cache_file}")
        try:
            rows = read_json_file(cache_file)
            if rows:
                # Convert to our standard format
                adapted_rows = []
                for row in rows:
                    try:
                        # Convert from BFCL format to our standard format
                        tools_json = orjson.dumps(
                            row.get("function", [])).decode('utf-8')
                        messages_json = orjson.dumps(
                            row.get("question", [[]])[0]).decode('utf-8')

                        # Create target from ground_truth
                        ground_truth_calls = row.get("ground_truth", [])
                        tool_calls = []
                        for call_str in ground_truth_calls:
                            # Parse function call string like "calculate_circle_area(radius=5.0)"
                            try:
                                import re
                                match = re.match(r'(\w+)\((.*)\)', call_str)
                                if match:
                                    func_name = match.group(1)
                                    args_str = match.group(2)

                                    # Parse arguments
                                    args = {}
                                    if args_str:
                                        # Simple parsing for key=value pairs
                                        for arg_pair in args_str.split(', '):
                                            if '=' in arg_pair:
                                                key, value = arg_pair.split(
                                                    '=', 1)
                                                # Try to evaluate the value
                                                try:
                                                    # Handle strings, numbers, lists
                                                    if value.startswith("'") and value.endswith("'"):
                                                        args[key] = value[1:-1]
                                                    elif value.startswith('[') and value.endswith(']'):
                                                        args[key] = eval(value)
                                                    else:
                                                        args[key] = float(
                                                            value) if '.' in value else int(value)
                                                except:
                                                    args[key] = value

                                    tool_calls.append({
                                        "name": func_name,
                                        "arguments": args
                                    })
                            except Exception:
                                # Skip malformed calls
                                continue

                        target_json = orjson.dumps(
                            {"tool_calls": tool_calls}).decode('utf-8')

                        adapted_row = {
                            "tools_json": tools_json,
                            "messages_json": messages_json,
                            "target_json": target_json,
                            "meta_source": "synthetic_parallel",
                            "n_calls": len(tool_calls),
                            "difficulty": "parallel" if len(set(call.get("name", "") for call in tool_calls)) > 1 else "multiple",
                            "valid": len(tool_calls) > 0
                        }
                        adapted_rows.append(adapted_row)

                    except Exception as e:
                        print(f"[warn] Failed to adapt synthetic row: {e}")
                        continue

                if adapted_rows:
                    ds = Dataset.from_list(adapted_rows)
                    ds = ds.filter(lambda ex: bool(ex["valid"]))

                    # Apply difficulty classification
                    ds = ds.map(add_difficulty)

                    if cap and len(ds) > cap:
                        ds = ds.shuffle(seed=RANDOM_SEED).select(range(cap))

                    load_time = time.time() - start_time
                    print(
                        f"[ok] Synthetic parallel data (cached): {len(ds)} examples loaded in {load_time:.2f}s")

                    # Print statistics
                    difficulties = {}
                    for ex in ds:
                        diff = ex.get("difficulty", "unknown")  # type: ignore
                        difficulties[diff] = difficulties.get(diff, 0) + 1
                    print("Difficulty distribution:")
                    for diff, count in difficulties.items():
                        print(
                            f"  {diff}: {count} examples ({count/len(ds)*100:.1f}%)")

                    return ds
        except Exception as e:
            print(f"[warn] Failed to load cached synthetic data: {e}")

    # Generate new synthetic data
    print("[generate] Generating new synthetic parallel data...")
    try:
        from src.synthetic_parallel_generator import SyntheticParallelGenerator

        # Initialize generator (will use HF_URL from environment if available)
        generator = SyntheticParallelGenerator()

        # Generate synthetic examples
        synthetic_examples = generator.generate_synthetic_examples(
            target_count=cap)

        # Validate generated examples
        validated_examples = generator.validate_data(synthetic_examples)

        if not validated_examples:
            print("[error] No valid synthetic examples generated")
            return None

        # Convert to our standard format
        adapted_rows = []
        for example in validated_examples:
            try:
                # Convert from BFCL format to our standard format
                tools_json = orjson.dumps(
                    example.get("function", [])).decode('utf-8')
                messages_json = orjson.dumps(example.get(
                    "question", [[]])[0]).decode('utf-8')

                # Target is already in the right format
                target_json = orjson.dumps({"tool_calls": [
                    {"name": call.split('(')[0], "arguments": {}} for call in example.get("ground_truth", [])
                ]}).decode('utf-8')

                adapted_row = {
                    "tools_json": tools_json,
                    "messages_json": messages_json,
                    "target_json": target_json,
                    "meta_source": "synthetic_parallel",
                    "n_calls": len(example.get("ground_truth", [])),
                    "difficulty": "parallel" if len(example.get("ground_truth", [])) > 1 else "simple",
                    "valid": True
                }
                adapted_rows.append(adapted_row)

            except Exception as e:
                print(f"[warn] Failed to adapt generated example: {e}")
                continue

        if not adapted_rows:
            print("[error] No examples could be adapted")
            return None

        # Save cache for future use
        try:
            import json as json_std
            with open(cache_file, 'w', encoding='utf-8') as f:
                json_std.dump(validated_examples, f,
                              indent=2, ensure_ascii=False)
            print(f"[cache] Saved synthetic data to {cache_file}")
        except Exception as e:
            print(f"[warn] Failed to save cache: {e}")

        # Create dataset
        ds = Dataset.from_list(adapted_rows)
        ds = ds.map(add_difficulty)

        if cap and len(ds) > cap:
            ds = ds.shuffle(seed=RANDOM_SEED).select(range(cap))

        generation_time = time.time() - start_time
        print(
            f"[ok] Synthetic parallel data (generated): {len(ds)} examples created in {generation_time:.2f}s")
        return ds

    except Exception as e:
        print(f"[error] Failed to generate synthetic data: {e}")
        return None


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
        # Use int32 instead of int64 for Parquet compatibility
        "n_calls": Value("int32"),
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
    from src.config import OPENFUNCTIONS_FILES, TOOLBENCH_FILES

    print("=" * 60)
    print("CREATING MIXED DATASET")
    print("=" * 60)
    start_time = time.time()

    parts = []

    # Load xLAM dataset
    print("\n[1/5] Loading xLAM dataset...")
    xlam_ds = load_xlam60k()
    if xlam_ds is not None and len(xlam_ds) > 0:
        parts.append(xlam_ds)
        print(f"[ok] xLAM: {len(xlam_ds)} examples")

    # Load OpenFunctions dataset
    print("\n[2/5] Loading OpenFunctions dataset...")
    openfunc_ds = load_openfunctions_local(OPENFUNCTIONS_FILES)
    if openfunc_ds is not None and len(openfunc_ds) > 0:
        parts.append(openfunc_ds)
        print(f"[ok] OpenFunctions: {len(openfunc_ds)} examples")

    # Load ToolBench dataset
    print("\n[3/5] Loading ToolBench dataset...")
    toolbench_ds = load_toolbench_local(TOOLBENCH_FILES)
    if toolbench_ds is not None and len(toolbench_ds) > 0:
        parts.append(toolbench_ds)
        print(f"[ok] ToolBench: {len(toolbench_ds)} examples")

    # Load Dolly no-call dataset
    print("\n[4/5] Loading Dolly no-call dataset...")
    dolly_nocall = load_dolly_nocall()
    if dolly_nocall is not None and len(dolly_nocall) > 0:
        parts.append(dolly_nocall)
        print(f"[ok] Dolly no-call: {len(dolly_nocall)} examples")

    # Load WikiText no-call dataset
    print("\n[5/6] Loading WikiText no-call dataset...")
    wtx = load_wikitext_nocall()
    if wtx is not None and len(wtx) > 0:
        parts.append(wtx)
        print(f"[ok] WikiText no-call: {len(wtx)} examples")

    # Load synthetic parallel dataset
    print("\n[6/6] Loading synthetic parallel dataset...")
    synthetic_ds = load_synthetic_parallel_data()
    if synthetic_ds is not None and len(synthetic_ds) > 0:
        parts.append(synthetic_ds)
        print(f"[ok] Synthetic parallel: {len(synthetic_ds)} examples")

    if not parts:
        raise RuntimeError("No data loaded. Check paths and try again.")

    print(f"\n[concatenate] Combining {len(parts)} datasets...")
    concat_start = time.time()
    mix = concatenate_datasets(parts).shuffle(seed=RANDOM_SEED)
    concat_time = time.time() - concat_start
    print(f"[ok] Concatenation completed in {concat_time:.2f}s")
    print(f"[ok] TOTAL adapted: {len(mix)} examples")

    # Ensure Hub-compatible schema
    print("\n[finalize] Ensuring Hub-compatible schema...")
    schema_start = time.time()
    mix = ensure_hub_compatible_schema(mix)
    schema_time = time.time() - schema_start
    print(f"[ok] Schema finalization completed in {schema_time:.2f}s")

    total_time = time.time() - start_time

    # Analyze final dataset composition
    print("\n[analysis] Dataset composition analysis...")
    source_counts = {}
    call_distribution = {}

    # Convert dataset to list for analysis to avoid Arrow type issues
    mix_list = list(mix)
    for ex in mix_list:
        source = getattr(ex, "meta_source", "unknown")
        n_calls = getattr(ex, "n_calls", 0)

        source_counts[source] = source_counts.get(source, 0) + 1
        call_distribution[n_calls] = call_distribution.get(n_calls, 0) + 1

    total_examples = len(mix)
    tool_calling_examples = sum(
        count for calls, count in call_distribution.items() if calls > 0)
    no_call_examples = call_distribution.get(0, 0)
    multi_call_examples = sum(
        count for calls, count in call_distribution.items() if calls > 1)

    print("=" * 60)
    print(f"FINAL DATASET ANALYSIS: {total_examples} examples")
    print("=" * 60)
    print("By source:")
    for source, count in sorted(source_counts.items()):
        pct = count / total_examples * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    print("\nBy function calling:")
    print(
        f"  Tool calling: {tool_calling_examples:,} ({tool_calling_examples/total_examples*100:.1f}%)")
    print(
        f"  No-call: {no_call_examples:,} ({no_call_examples/total_examples*100:.1f}%)")
    print(
        f"  Multi-call: {multi_call_examples:,} ({multi_call_examples/total_examples*100:.1f}%)")

    print("\nCall count distribution:")
    for calls in sorted(call_distribution.keys()):
        count = call_distribution[calls]
        pct = count / total_examples * 100
        if calls == 0:
            print(f"  {calls} calls (no-call): {count:,} ({pct:.1f}%)")
        elif calls == 1:
            print(f"  {calls} call: {count:,} ({pct:.1f}%)")
        else:
            print(f"  {calls} calls: {count:,} ({pct:.1f}%)")

    print(f"\nTotal processing time: {total_time:.2f}s")
    print("=" * 60)

    return mix
