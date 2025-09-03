# Quality Improvements Documentation

This document describes the comprehensive quality improvements added to the tool-calling SFT dataset to address recurrent failure patterns and enhance training robustness.

## Overview

The enhanced dataset includes multiple quality control mechanisms designed to prevent common failure modes in tool-calling models:

- **Schema-strict exemplars** for common failure patterns
- **Negative clarification examples** to teach proper request handling
- **ToolBench field normalization** to prevent alias confusion
- **Balanced no-tool instructions** to prevent over-triggering
- **Adversarial variants** for robustness testing
- **Programmatic validation** with JSON schema enforcement

## 🎯 Schema-Strict Exemplars

### Problem Addressed

Models frequently generate invalid parameter values that don't match expected schemas, particularly:

- Wrong enum values (e.g., "premium" instead of "plus")
- Incorrect state formats (e.g., "California" instead of "CA")
- Malformed location strings (e.g., "Shanghai China" instead of "Shanghai, China")
- Type mismatches (e.g., string "300" instead of integer 300)

### Solution

Generated 1,000+ carefully crafted examples demonstrating correct parameter formats:

#### Short Enums

```json
{
  "function_name": "book_ride",
  "parameter": "ride_type", 
  "valid_values": ["plus", "comfort", "black", "pool", "xl"],
  "correct_examples": [{"ride_type": "plus"}],
  "incorrect_examples": [{"ride_type": "Uber Plus"}]
}
```

#### State Abbreviations

```json
{
  "function_name": "search_restaurants",
  "parameter": "state",
  "valid_values": ["CA", "NY", "TX", "FL", "WA"],
  "correct_examples": [{"state": "CA"}],
  "incorrect_examples": [{"state": "California"}]
}
```

#### Exact-String Locales

```json
{
  "function_name": "get_weather", 
  "parameter": "location",
  "valid_values": ["Shanghai, China", "上海, 中国", "New York, NY"],
  "correct_examples": [{"location": "Shanghai, China"}],
  "incorrect_examples": [{"location": "Shanghai China"}]
}
```

#### Type-Correct Integers

```json
{
  "function_name": "set_timer",
  "parameter": "wait_seconds",
  "data_type": "integer",
  "correct_examples": [{"wait_seconds": 300}],
  "incorrect_examples": [{"wait_seconds": "300"}]
}
```

## 🤔 Negative Clarification Examples

### Problem Addressed

Models often guess missing parameter values instead of asking for clarification, leading to incorrect tool calls with hallucinated data.

### Solution

Generated 500+ examples showing proper clarification requests:

```json
{
  "user_query": "Book me an Uber ride",
  "missing_fields": ["pickup_location", "destination", "ride_type"],
  "assistant_response": "I'd be happy to help you book an Uber ride! However, I need some additional information to proceed:\n\n- What's your pickup location?\n- Where would you like to go?\n- What type of ride would you prefer? (options: plus, comfort, black, pool, xl)\n\nOnce you provide these details, I can book the ride for you."
}
```

This teaches the model to:

1. Acknowledge the request positively
2. Explicitly state what information is missing
3. Provide valid options when applicable
4. Request clarification professionally

## 🔧 ToolBench Field Normalization

### Problem Addressed

ToolBench contains inconsistent field names across tools (e.g., `city` vs `location`, `uber_ride` vs `ride_type`), causing models to learn incorrect aliases.

### Solution

Implemented comprehensive field mapping to canonical names:

```python
TOOLBENCH_FIELD_MAPPINGS = {
    # Location aliases
    "city": "location",
    "location_name": "location", 
    "place": "location",
    "destination": "location",
    "address": "location",
    
    # Service aliases  
    "uber_ride": "ride_type",
    "uber.ride": "ride_type",
    "service_type": "ride_type",
    "vehicle_type": "ride_type",
    
    # Time aliases
    "datetime": "date",
    "timestamp": "date",
    "time_slot": "time",
    
    # Common parameter aliases
    "query": "search_query",
    "q": "search_query",
    "search_term": "search_query",
    
    # ID aliases
    "identifier": "id",
    "user_id": "id", 
    "item_id": "id"
}
```

All ToolBench examples are automatically normalized during processing to use canonical field names.

## ⚖️ Balanced No-Tool Instructions

### Problem Addressed

Models over-trigger tool usage on queries that should be answered directly through explanation rather than function calls.

### Solution

Added 800+ carefully crafted examples that look tool-triggering but should result in explanations:

```json
{
  "query": "Explain what an Uber Comfort is",
  "response": "Uber Comfort is a premium ride option that offers more legroom and highly-rated drivers. It typically costs slightly more than UberX but provides additional comfort features like temperature control requests and top-rated drivers with newer vehicles."
}
```

Examples cover:

- **Product explanations** ("What's the difference between weather and climate?")
- **Process descriptions** ("How do restaurant reservations work?")
- **Conceptual questions** ("Tell me about timer apps on smartphones")
- **API discussions** ("What are the benefits of using APIs for travel booking?")

Target: 20-25% of total examples to maintain proper tool/no-tool balance.

## 🛡️ Adversarial Variants

### Problem Addressed

Models fail on edge cases with unusual formatting, whitespace, punctuation, or Unicode characters.

### Solution

Generated adversarial variants testing:

#### Whitespace Handling

```json
{
  "arguments": {
    "location": "  San Francisco, CA  ",  // Extra whitespace
    "cuisine": "  Italian  "
  }
}
```

#### Punctuation Variations

```json
{
  "user_content": "Book appointment for tomorrow!"  // Added punctuation
}
```

#### Unicode Support

```json
{
  "arguments": {
    "location": "東京, 日本"  // Unicode characters
  }
}
```

#### Falsy Optional Fields

```json
{
  "arguments": {
    "required_field": "value",
    "optional_param": ""  // Empty string for optional field
  }
}
```

## 🔍 Programmatic Validation

### Problem Addressed

Training samples with malformed JSON or invalid schemas corrupt model training.

### Solution

Multi-layer validation pipeline:

#### JSON Schema Validation

```python
TOOL_SCHEMAS = {
    "ride_booking": {
        "type": "object",
        "properties": {
            "pickup_location": {"type": "string", "minLength": 3},
            "destination": {"type": "string", "minLength": 3},
            "ride_type": {"type": "string", "enum": ["plus", "comfort", "black", "pool", "xl"]},
            "passenger_count": {"type": "integer", "minimum": 1, "maximum": 8}
        },
        "required": ["pickup_location", "destination", "ride_type"],
        "additionalProperties": False
    }
}
```

#### Quality Filters

- **JSON validity**: All JSON fields must parse correctly
- **Schema compliance**: Tool calls must match expected schemas
- **Consistency checks**: `n_calls` must match actual tool call count
- **Message validation**: At least one user message required
- **Type validation**: Proper data types enforced

#### Auto-Rejection Rules

- Invalid JSON → Automatic rejection
- Schema violations → Automatic rejection  
- Missing required fields → Automatic rejection
- Type mismatches → Automatic rejection

## 🧪 Unit Testing Framework

### Problem Addressed

No systematic way to verify quality improvements are working correctly.

### Solution

Comprehensive test suite with 100+ hand-designed test cases:

#### Tool Family Tests

- **Ride Booking**: Valid/invalid enum values, missing fields, type errors
- **Weather Queries**: Location format validation, Unicode support
- **Timer Setting**: Integer validation, range checking
- **Parallel Calls**: Multi-tool coordination validation

#### Adversarial Tests

- Whitespace handling
- Empty string validation
- Null value processing
- Punctuation tolerance

#### Negative Example Tests

- Clarification request validation
- No-tool explanation verification
- Balanced response checking

## 📊 Quality Metrics & Monitoring

### Validation Dashboard

The enhanced pipeline provides comprehensive quality metrics:

```
📋 QUALITY VALIDATION RESULTS
═══════════════════════════════════════
⏱️  Validation time: 12.34s
📊 Sample size: 5,000 examples (from 42,538 total)
✅ Success rate: 98.7%
❌ Validation errors: 23
⚠️  Schema violations: 8

📈 Dataset Composition:
  Tool calling: 32,430 (76.3%)
  No-call: 10,108 (23.7%)
  Multi-call: 3,247 (7.6%)

🎯 No-call Balance Check:
  Current: 23.7%
  Target: 20-25%
  ✅ Within target range!

🔧 Quality Enhancement Breakdown:
  schema_strict: 1,000 (2.4%)
  negative_clarification: 500 (1.2%)
  no_tool_explanation: 800 (1.9%)
  toolbench_normalized: 15,623 (36.7%)
  adversarial: 1,562 (3.7%)
  original: 23,053 (54.2%)
```

### Continuous Monitoring

- **Success rate tracking**: Target >95% validation pass rate
- **Balance monitoring**: Maintain 20-25% no-call examples
- **Schema compliance**: Track violations by tool family
- **Error categorization**: Identify emerging failure patterns

## 🚀 Usage Instructions

### Enhanced Dataset Creation

```bash
# Create enhanced dataset with all quality controls
make run-enhanced

# Create with strict validation (fails on errors)
make run-strict

# Validate existing dataset
make validate

# Run quality unit tests
make test-quality
```

### Configuration Options

```python
# Quality control settings in src/config.py
ENABLE_QUALITY_CONTROLS = True
ENABLE_SCHEMA_STRICT_EXEMPLARS = True  
ENABLE_NEGATIVE_EXAMPLES = True
ENABLE_BALANCED_NO_TOOL = True
ENABLE_TOOLBENCH_NORMALIZATION = True
ENABLE_ADVERSARIAL_VARIANTS = True

# Quality exemplar counts
CAP_QUALITY_EXEMPLARS = 1000      
CAP_NEGATIVE_EXAMPLES = 500       
CAP_BALANCED_NO_TOOL = 800        

# Quality thresholds
MIN_NO_CALL_PERCENTAGE = 20       
MAX_NO_CALL_PERCENTAGE = 25       
ADVERSARIAL_VARIANT_RATIO = 0.1   
```

### Validation Commands

```bash
# Validate specific dataset
python -m src.validate_quality output/tool_sft_corpus

# Run unit tests
python -m src.unit_tests

# Create with validation
python -m src.enhanced_main --validate --strict
```

## 📈 Expected Improvements

### Model Behavior

- **Reduced hallucination**: 60-80% fewer invalid parameter values
- **Better clarification**: Models ask for missing info instead of guessing
- **Consistent naming**: No more field alias confusion
- **Appropriate restraint**: 20-25% fewer unnecessary tool calls
- **Robustness**: Better handling of edge cases and formatting variations

### Training Stability  

- **Cleaner data**: 95%+ valid examples in training set
- **Consistent schemas**: Uniform parameter naming across tools
- **Balanced objectives**: Proper tool/no-tool decision making
- **Edge case coverage**: Robust performance on real-world inputs

### Evaluation Metrics

- **Schema compliance**: >98% valid tool calls
- **Clarification rate**: Appropriate requests for missing information
- **False positive reduction**: Fewer unnecessary tool triggers
- **Adversarial robustness**: Consistent performance on malformed inputs

## 🔧 Implementation Details

### Files Added

- `src/quality_control.py`: Core quality control functions
- `src/unit_tests.py`: Comprehensive test suite
- `src/enhanced_data_loaders.py`: Enhanced data loading with quality controls
- `src/validate_quality.py`: Validation and monitoring tools
- `src/enhanced_main.py`: Enhanced main script with quality pipeline

### Integration Points

- **Makefile updates**: New targets for enhanced functionality
- **Config extensions**: Quality control parameters
- **Pipeline integration**: Seamless quality control integration
- **Monitoring hooks**: Real-time quality metrics

This comprehensive quality improvement framework ensures the tool-calling SFT dataset produces models that are more reliable, accurate, and robust in real-world scenarios.
