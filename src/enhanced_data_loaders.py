"""Enhanced data loading functions with quality controls and exemplar generation."""

import os
import time
import orjson
import random
from datasets import Dataset, concatenate_datasets
from typing import Optional
from tqdm import tqdm

from src.config import (
    USE_TOOLBENCH, CAP_TOOLBENCH, RANDOM_SEED
)
from src.parsers import adapt_toolbench_row
from src.utils import make_empty_row, read_json_file, add_difficulty
from src.quality_control import (
    generate_schema_strict_example,
    generate_negative_example, 
    generate_no_tool_example,
    create_adversarial_variant,
    normalize_toolbench_fields,
    filter_quality_examples,
    CLARIFICATION_TEMPLATES,
    NO_TOOL_PATTERNS,
    SCHEMA_STRICT_EXEMPLARS
)


def generate_quality_exemplars(target_count: int = 1000) -> Optional[Dataset]:
    """Generate high-quality schema-strict exemplars for common failure patterns."""
    print(f"[start] Generating {target_count} quality exemplars...")
    start_time = time.time()
    
    exemplars = []
    patterns = list(SCHEMA_STRICT_EXEMPLARS.keys())
    
    # Distribute target count across pattern types
    per_pattern = target_count // len(patterns)
    remainder = target_count % len(patterns)
    
    for i, pattern_type in enumerate(patterns):
        count = per_pattern + (1 if i < remainder else 0)
        print(f"[generate] Creating {count} examples for {pattern_type}")
        
        for _ in range(count):
            try:
                example = generate_schema_strict_example(pattern_type, {})
                if example and example.get("valid"):
                    exemplars.append(example)
            except Exception as e:
                print(f"[warn] Failed to generate {pattern_type} example: {e}")
                continue
    
    if not exemplars:
        print("[error] No quality exemplars generated")
        return None
    
    # Create dataset
    ds = Dataset.from_list(exemplars)
    ds = ds.map(add_difficulty)
    
    generation_time = time.time() - start_time
    print(f"[ok] Quality exemplars: {len(ds)} examples generated in {generation_time:.2f}s")
    
    return ds


def generate_negative_clarification_examples(target_count: int = 500) -> Optional[Dataset]:
    """Generate negative examples showing clarification requests instead of guessing."""
    print(f"[start] Generating {target_count} negative clarification examples...")
    start_time = time.time()
    
    examples = []
    templates = CLARIFICATION_TEMPLATES
    
    # Generate examples from templates
    per_template = target_count // len(templates)
    remainder = target_count % len(templates)
    
    for i, template in enumerate(templates):
        count = per_template + (1 if i < remainder else 0)
        
        for _ in range(count):
            try:
                example = generate_negative_example(template)
                if example and example.get("valid"):
                    examples.append(example)
            except Exception as e:
                print(f"[warn] Failed to generate negative example: {e}")
                continue
    
    if not examples:
        print("[error] No negative examples generated")
        return None
    
    # Create dataset
    ds = Dataset.from_list(examples)
    ds = ds.map(add_difficulty)
    
    generation_time = time.time() - start_time
    print(f"[ok] Negative clarification examples: {len(ds)} examples generated in {generation_time:.2f}s")
    
    return ds


def generate_balanced_no_tool_examples(target_count: int = 800) -> Optional[Dataset]:
    """Generate balanced no-tool examples that look tool-triggering but aren't."""
    print(f"[start] Generating {target_count} balanced no-tool examples...")
    start_time = time.time()
    
    examples = []
    patterns = NO_TOOL_PATTERNS
    
    # Generate examples from patterns
    per_pattern = target_count // len(patterns)
    remainder = target_count % len(patterns)
    
    for i, pattern in enumerate(patterns):
        count = per_pattern + (1 if i < remainder else 0)
        
        for _ in range(count):
            try:
                example = generate_no_tool_example(pattern)
                if example and example.get("valid"):
                    examples.append(example)
            except Exception as e:
                print(f"[warn] Failed to generate no-tool example: {e}")
                continue
    
    if not examples:
        print("[error] No no-tool examples generated")
        return None
    
    # Create dataset
    ds = Dataset.from_list(examples)
    ds = ds.map(add_difficulty)
    
    generation_time = time.time() - start_time
    print(f"[ok] Balanced no-tool examples: {len(ds)} examples generated in {generation_time:.2f}s")
    
    return ds


def enhance_toolbench_with_normalization(files: list[str], cap: int = CAP_TOOLBENCH) -> Optional[Dataset]:
    """Load ToolBench with field normalization and quality filtering."""
    if not USE_TOOLBENCH:
        print("[skip] ToolBench disabled in config")
        return None
    
    print(f"[start] Loading enhanced ToolBench from {len(files)} files...")
    start_time = time.time()
    
    # Load raw data first (reusing existing logic)
    all_rows = []
    print("[step 1/5] Reading JSON files...")
    
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
            
        all_rows.extend(rows)
        file_size = os.path.getsize(fp) / (1024 * 1024)  # MB
        print(f"[ok] loaded {len(rows)} rows from {os.path.basename(fp)} ({file_size:.1f}MB, {file_time:.2f}s)")
    
    if not all_rows:
        print("[error] No ToolBench data loaded")
        return None

    print(f"[step 2/5] Creating dataset from {len(all_rows)} rows...")
    dataset_start = time.time()
    
    # Create dataset with single uniform column
    ds = Dataset.from_dict({"raw_json": [orjson.dumps(r).decode('utf-8') for r in all_rows]})
    dataset_time = time.time() - dataset_start
    print(f"[ok] Dataset created in {dataset_time:.2f}s")

    # Map: parse, adapt, normalize, drop raw_json
    print("[step 3/5] Adapting and normalizing ToolBench rows...")
    adapt_start = time.time()
    
    def _map_with_normalization(row):
        try:
            obj = orjson.loads(row["raw_json"].encode('utf-8'))
        except Exception:
            return make_empty_row()
        
        # First adapt using existing logic
        adapted = adapt_toolbench_row(obj)
        
        # Then apply normalization to tool calls
        if adapted.get("valid") and adapted.get("target_json"):
            try:
                target = orjson.loads(adapted["target_json"].encode('utf-8'))
                if "tool_calls" in target:
                    normalized_calls = []
                    for call in target["tool_calls"]:
                        normalized_call = normalize_toolbench_fields(call)
                        normalized_calls.append(normalized_call)
                    
                    target["tool_calls"] = normalized_calls
                    adapted["target_json"] = orjson.dumps(target).decode('utf-8')
                    adapted["meta_source"] = "toolbench_normalized"
            except Exception:
                # If normalization fails, keep original
                pass
        
        return adapted

    mapped = ds.map(_map_with_normalization, remove_columns=["raw_json"], desc="Adapting ToolBench with normalization")
    adapt_time = time.time() - adapt_start
    print(f"[ok] Adaptation and normalization completed in {adapt_time:.2f}s")
    
    print("[step 4/5] Quality filtering...")
    filter_start = time.time()
    
    # Filter for valid examples
    mapped = mapped.filter(lambda ex: bool(ex["valid"]))
    
    # Apply quality filtering
    print("[step 4a/5] Converting to list for quality filtering...")
    example_list = [dict(ex) for ex in mapped]  # Convert to list of dicts
    print(f"[step 4b/5] Applying quality filters to {len(example_list)} examples...")
    filtered_list = filter_quality_examples(example_list)
    print(f"[ok] Quality filtering retained {len(filtered_list)}/{len(example_list)} examples")
    
    # Convert back to dataset
    if filtered_list:
        mapped = Dataset.from_list(filtered_list)
    else:
        print("[warn] No examples passed quality filtering")
        return None
    
    mapped = mapped.map(add_difficulty)
    
    if cap and len(mapped) > cap:
        print(f"[cap] Limiting to {cap} examples (from {len(mapped)})")
        mapped = mapped.shuffle(seed=RANDOM_SEED).select(range(cap))
    
    filter_time = time.time() - filter_start
    
    print("[step 5/5] Creating adversarial variants...")
    adversarial_start = time.time()
    
    # Create adversarial variants for a subset (10% of examples)
    adversarial_count = min(len(mapped) // 10, 1000)
    if adversarial_count > 0:
        adversarial_indices = random.sample(range(len(mapped)), adversarial_count)
        adversarial_examples = []
        
        for idx in adversarial_indices:
            try:
                original = mapped[idx]
                variant = create_adversarial_variant(dict(original))
                if variant and variant != original:
                    adversarial_examples.append(variant)
            except Exception:
                continue
        
        if adversarial_examples:
            adversarial_ds = Dataset.from_list(adversarial_examples)
            mapped = concatenate_datasets([mapped, adversarial_ds])
            print(f"[ok] Added {len(adversarial_examples)} adversarial variants")
    
    adversarial_time = time.time() - adversarial_start
    total_time = time.time() - start_time
    
    print(f"[ok] Enhanced ToolBench: {len(mapped)} examples loaded in {total_time:.2f}s total")
    print(f"    - Dataset creation: {dataset_time:.2f}s") 
    print(f"    - Adaptation + normalization: {adapt_time:.2f}s")
    print(f"    - Quality filtering: {filter_time:.2f}s")
    print(f"    - Adversarial variants: {adversarial_time:.2f}s")
    
    return mapped


def create_enhanced_mixed_dataset() -> Dataset:
    """Create enhanced mixed dataset with quality controls and exemplar generation."""
    from src.config import OPENFUNCTIONS_FILES, TOOLBENCH_FILES
    from src.data_loaders import load_xlam60k, load_openfunctions_local, load_dolly_nocall, load_wikitext_nocall
    
    print("=" * 60)
    print("CREATING ENHANCED MIXED DATASET WITH QUALITY CONTROLS")
    print("=" * 60)
    start_time = time.time()
    
    parts = []

    # Load existing datasets (using existing loaders for now)
    print("\n[1/8] Loading xLAM dataset...")
    xlam_ds = load_xlam60k()
    if xlam_ds is not None and len(xlam_ds) > 0:
        parts.append(xlam_ds)
        print(f"[ok] xLAM: {len(xlam_ds)} examples")

    print("\n[2/8] Loading OpenFunctions dataset...")
    openfunc_ds = load_openfunctions_local(OPENFUNCTIONS_FILES)
    if openfunc_ds is not None and len(openfunc_ds) > 0:
        parts.append(openfunc_ds)
        print(f"[ok] OpenFunctions: {len(openfunc_ds)} examples")

    print("\n[3/8] Loading enhanced ToolBench dataset...")
    toolbench_ds = enhance_toolbench_with_normalization(TOOLBENCH_FILES)
    if toolbench_ds is not None and len(toolbench_ds) > 0:
        parts.append(toolbench_ds)
        print(f"[ok] Enhanced ToolBench: {len(toolbench_ds)} examples")

    print("\n[4/8] Loading Dolly no-call dataset...")
    dolly_nocall = load_dolly_nocall()
    if dolly_nocall is not None and len(dolly_nocall) > 0:
        parts.append(dolly_nocall)
        print(f"[ok] Dolly no-call: {len(dolly_nocall)} examples")

    print("\n[5/8] Loading WikiText no-call dataset...")
    wtx = load_wikitext_nocall()
    if wtx is not None and len(wtx) > 0:
        parts.append(wtx)
        print(f"[ok] WikiText no-call: {len(wtx)} examples")

    # Generate quality exemplars
    print("\n[6/8] Generating quality exemplars...")
    quality_exemplars = generate_quality_exemplars(1000)
    if quality_exemplars is not None and len(quality_exemplars) > 0:
        parts.append(quality_exemplars)
        print(f"[ok] Quality exemplars: {len(quality_exemplars)} examples")

    # Generate negative clarification examples  
    print("\n[7/8] Generating negative clarification examples...")
    negative_examples = generate_negative_clarification_examples(500)
    if negative_examples is not None and len(negative_examples) > 0:
        parts.append(negative_examples)
        print(f"[ok] Negative examples: {len(negative_examples)} examples")

    # Generate balanced no-tool examples
    print("\n[8/8] Generating balanced no-tool examples...")
    no_tool_examples = generate_balanced_no_tool_examples(800)
    if no_tool_examples is not None and len(no_tool_examples) > 0:
        parts.append(no_tool_examples)
        print(f"[ok] No-tool examples: {len(no_tool_examples)} examples")

    if not parts:
        raise RuntimeError("No data loaded. Check paths and try again.")

    print(f"\n[concatenate] Combining {len(parts)} datasets...")
    concat_start = time.time()
    mix = concatenate_datasets(parts).shuffle(seed=RANDOM_SEED)
    concat_time = time.time() - concat_start
    print(f"[ok] Concatenation completed in {concat_time:.2f}s")
    print(f"[ok] TOTAL enhanced examples: {len(mix)}")
    
    total_time = time.time() - start_time
    
    # Analyze final dataset composition with enhanced categories
    print("\n[analysis] Enhanced dataset composition analysis...")
    source_counts = {}
    call_distribution = {}
    quality_categories = {
        "schema_strict": 0,
        "negative_clarification": 0, 
        "no_tool_explanation": 0,
        "toolbench_normalized": 0,
        "adversarial": 0,
        "original": 0
    }
    
    # Convert dataset to list for analysis
    mix_list = list(mix)
    for ex in mix_list:
        source = getattr(ex, "meta_source", "unknown")
        n_calls = getattr(ex, "n_calls", 0)
        
        source_counts[source] = source_counts.get(source, 0) + 1
        call_distribution[n_calls] = call_distribution.get(n_calls, 0) + 1
        
        # Categorize by quality enhancement type
        if "schema_strict" in source:
            quality_categories["schema_strict"] += 1
        elif "negative_clarification" in source:
            quality_categories["negative_clarification"] += 1
        elif "no_tool_explanation" in source:
            quality_categories["no_tool_explanation"] += 1
        elif "toolbench_normalized" in source:
            quality_categories["toolbench_normalized"] += 1
        elif "adversarial" in source:
            quality_categories["adversarial"] += 1
        else:
            quality_categories["original"] += 1
    
    total_examples = len(mix)
    tool_calling_examples = sum(count for calls, count in call_distribution.items() if calls > 0)
    no_call_examples = call_distribution.get(0, 0)
    multi_call_examples = sum(count for calls, count in call_distribution.items() if calls > 1)
    
    print("=" * 60)
    print(f"ENHANCED DATASET ANALYSIS: {total_examples} examples")
    print("=" * 60)
    print("By source:")
    for source, count in sorted(source_counts.items()):
        pct = count / total_examples * 100
        print(f"  {source}: {count:,} ({pct:.1f}%)")
    
    print("\nBy quality enhancement:")
    for category, count in quality_categories.items():
        pct = count / total_examples * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")
    
    print("\nBy function calling:")
    print(f"  Tool calling: {tool_calling_examples:,} ({tool_calling_examples/total_examples*100:.1f}%)")
    print(f"  No-call: {no_call_examples:,} ({no_call_examples/total_examples*100:.1f}%)")
    print(f"  Multi-call: {multi_call_examples:,} ({multi_call_examples/total_examples*100:.1f}%)")
    
    no_call_percentage = no_call_examples / total_examples * 100
    print(f"\n📊 No-call balance: {no_call_percentage:.1f}% (target: 20-25%)")
    if 20 <= no_call_percentage <= 25:
        print("✅ No-call balance is within target range!")
    elif no_call_percentage < 20:
        print("⚠️  No-call percentage is below target (need more no-call examples)")
    else:
        print("⚠️  No-call percentage is above target (need more tool-calling examples)")
    
    print(f"\nTotal processing time: {total_time:.2f}s")
    print("=" * 60)
    
    return mix
