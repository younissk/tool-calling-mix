"""Quality validation script for the enhanced tool-calling dataset."""

import json
import time
from typing import Dict, Any
from datasets import Dataset, load_from_disk
from src.quality_control import validate_example_json_schema, validate_tool_call_schema
from src.unit_tests import run_manual_test_suite


def validate_dataset_quality(dataset: Dataset) -> Dict[str, Any]:
    """Comprehensive quality validation of the dataset."""
    print("🔍 Running comprehensive dataset quality validation...")
    start_time = time.time()
    
    validation_results = {
        "total_examples": len(dataset),
        "validation_errors": [],
        "schema_violations": [],
        "quality_stats": {},
        "source_breakdown": {},
        "success_rate": 0.0
    }
    
    # Track statistics
    valid_count = 0
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
    
    print(f"📊 Validating {len(dataset)} examples...")
    
    # Sample validation for large datasets (validate first 10k examples)
    sample_size = min(len(dataset), 10000)
    sample_indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))[:sample_size]
    
    for i, idx in enumerate(sample_indices):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(sample_indices)} examples validated")
        
        try:
            example = dataset[idx]
            
            # Basic JSON validation
            is_valid, errors = validate_example_json_schema(example)
            if not is_valid:
                validation_results["validation_errors"].append({
                    "index": idx,
                    "errors": errors
                })
                continue
            
            valid_count += 1
            
            # Track source and call distribution
            source = example.get("meta_source", "unknown")
            n_calls = example.get("n_calls", 0)
            
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
            
            # Validate tool call schemas
            if n_calls > 0:
                try:
                    target = json.loads(example["target_json"])
                    for call in target.get("tool_calls", []):
                        call_name = call.get("name", "")
                        # Map to schema categories
                        if any(keyword in call_name.lower() for keyword in ["ride", "book", "uber"]):
                            schema_valid, error = validate_tool_call_schema(call, "ride_booking")
                            if not schema_valid:
                                validation_results["schema_violations"].append({
                                    "index": idx,
                                    "call_name": call_name,
                                    "schema": "ride_booking",
                                    "error": error
                                })
                        elif any(keyword in call_name.lower() for keyword in ["weather", "climate"]):
                            schema_valid, error = validate_tool_call_schema(call, "weather_query")
                            if not schema_valid:
                                validation_results["schema_violations"].append({
                                    "index": idx,
                                    "call_name": call_name,
                                    "schema": "weather_query",
                                    "error": error
                                })
                        elif any(keyword in call_name.lower() for keyword in ["timer", "alarm"]):
                            schema_valid, error = validate_tool_call_schema(call, "timer_setting")
                            if not schema_valid:
                                validation_results["schema_violations"].append({
                                    "index": idx,
                                    "call_name": call_name,
                                    "schema": "timer_setting",
                                    "error": error
                                })
                except Exception as e:
                    validation_results["validation_errors"].append({
                        "index": idx,
                        "errors": [f"Failed to parse target_json: {e}"]
                    })
        
        except Exception as e:
            validation_results["validation_errors"].append({
                "index": idx,
                "errors": [f"Unexpected error: {e}"]
            })
    
    # Calculate statistics
    total_examples = len(sample_indices)
    tool_calling_examples = sum(count for calls, count in call_distribution.items() if calls > 0)
    no_call_examples = call_distribution.get(0, 0)
    multi_call_examples = sum(count for calls, count in call_distribution.items() if calls > 1)
    
    validation_results["success_rate"] = valid_count / total_examples * 100
    validation_results["quality_stats"] = {
        "valid_examples": valid_count,
        "tool_calling_examples": tool_calling_examples,
        "no_call_examples": no_call_examples,
        "multi_call_examples": multi_call_examples,
        "no_call_percentage": no_call_examples / total_examples * 100,
        "validation_errors": len(validation_results["validation_errors"]),
        "schema_violations": len(validation_results["schema_violations"])
    }
    validation_results["source_breakdown"] = source_counts
    validation_results["quality_categories"] = quality_categories
    validation_results["call_distribution"] = call_distribution
    
    validation_time = time.time() - start_time
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("📋 QUALITY VALIDATION RESULTS")
    print("=" * 60)
    print(f"⏱️  Validation time: {validation_time:.2f}s")
    print(f"📊 Sample size: {total_examples:,} examples (from {len(dataset):,} total)")
    print(f"✅ Success rate: {validation_results['success_rate']:.1f}%")
    print(f"❌ Validation errors: {len(validation_results['validation_errors'])}")
    print(f"⚠️  Schema violations: {len(validation_results['schema_violations'])}")
    
    print("\n📈 Dataset Composition:")
    print(f"  Tool calling: {tool_calling_examples:,} ({tool_calling_examples/total_examples*100:.1f}%)")
    print(f"  No-call: {no_call_examples:,} ({no_call_examples/total_examples*100:.1f}%)")
    print(f"  Multi-call: {multi_call_examples:,} ({multi_call_examples/total_examples*100:.1f}%)")
    
    no_call_pct = no_call_examples / total_examples * 100
    print("\n🎯 No-call Balance Check:")
    print(f"  Current: {no_call_pct:.1f}%")
    print("  Target: 20-25%")
    if 20 <= no_call_pct <= 25:
        print("  ✅ Within target range!")
    elif no_call_pct < 20:
        print("  ⚠️  Below target - need more no-call examples")
    else:
        print("  ⚠️  Above target - need more tool-calling examples")
    
    print("\n🔧 Quality Enhancement Breakdown:")
    for category, count in quality_categories.items():
        pct = count / total_examples * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")
    
    if validation_results["validation_errors"]:
        print("\n❌ Top Validation Errors:")
        error_samples = validation_results["validation_errors"][:5]
        for i, error in enumerate(error_samples, 1):
            print(f"  {i}. Index {error['index']}: {error['errors'][:2]}")
    
    if validation_results["schema_violations"]:
        print("\n⚠️  Top Schema Violations:")
        violation_samples = validation_results["schema_violations"][:5]
        for i, violation in enumerate(violation_samples, 1):
            print(f"  {i}. {violation['call_name']} ({violation['schema']}): {violation['error'][:100]}")
    
    print("=" * 60)
    
    return validation_results


def run_unit_tests() -> Dict[str, Any]:
    """Run the comprehensive unit test suite."""
    print("\n🧪 Running unit test suite...")
    test_results = run_manual_test_suite()
    
    print("\n📋 UNIT TEST RESULTS:")
    print(f"  Total tests: {test_results['total_tests']}")
    print(f"  Passed: {test_results['passed']}")
    print(f"  Failed: {test_results['failed']}")
    print(f"  Success rate: {test_results['passed']/test_results['total_tests']*100:.1f}%")
    
    if test_results['failures']:
        print("\n❌ Test Failures:")
        for failure in test_results['failures'][:5]:
            print(f"  {failure['family']}/{failure['test']}")
    
    return test_results


def validate_enhanced_dataset(dataset_path: str = "output/tool_sft_corpus") -> Dict[str, Any]:
    """Main validation function for the enhanced dataset."""
    print("🚀 Starting comprehensive quality validation...")
    
    try:
        # Load dataset
        print(f"📂 Loading dataset from {dataset_path}...")
        if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
            from datasets import load_dataset
            dataset_raw = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset_raw = load_from_disk(dataset_path)
        
        # If it's a DatasetDict, use the train split
        if hasattr(dataset_raw, 'keys'):
            dataset = dataset_raw['train']
        else:
            dataset = dataset_raw
        
        print(f"✅ Loaded {len(dataset)} examples") # type: ignore
        
        # Run validations
        validation_results = validate_dataset_quality(dataset) # type: ignore
        unit_test_results = run_unit_tests()
        
        # Combine results
        combined_results = {
            "dataset_validation": validation_results,
            "unit_tests": unit_test_results,
            "overall_success": (
                validation_results["success_rate"] > 95.0 and 
                unit_test_results["passed"] / unit_test_results["total_tests"] > 0.9
            )
        }
        
        print("\n🎯 OVERALL ASSESSMENT:")
        if combined_results["overall_success"]:
            print("✅ Dataset passes quality validation!")
        else:
            print("❌ Dataset has quality issues that need attention")
        
        return combined_results
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "output/tool_sft_corpus"
    results = validate_enhanced_dataset(dataset_path)
    
    # Exit with error code if validation failed
    if not results.get("overall_success", False):
        sys.exit(1)
