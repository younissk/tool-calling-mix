"""Unit tests for tool calling quality control and validation."""

import json
from typing import Dict, Any
from src.quality_control import (
    validate_example_json_schema, 
    validate_tool_call_schema,
    normalize_toolbench_fields,
    generate_schema_strict_example,
    generate_negative_example,
    generate_no_tool_example,
    CLARIFICATION_TEMPLATES,
    NO_TOOL_PATTERNS
)


# Hand-designed test cases for common tool families
TOOL_FAMILY_TEST_CASES = {
    "ride_booking": [
        {
            "name": "Valid basic ride booking",
            "example": {
                "tools_json": json.dumps([{
                    "name": "book_ride",
                    "description": "Book a ride",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pickup_location": {"type": "string"},
                            "destination": {"type": "string"},
                            "ride_type": {"type": "string", "enum": ["plus", "comfort", "black", "pool", "xl"]}
                        },
                        "required": ["pickup_location", "destination", "ride_type"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Book me an Uber from downtown to the airport"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "book_ride",
                        "arguments": {
                            "pickup_location": "Downtown",
                            "destination": "Airport",
                            "ride_type": "plus"
                        }
                    }]
                }),
                "meta_source": "test_ride_booking",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": True
        },
        {
            "name": "Invalid ride type enum",
            "example": {
                "tools_json": json.dumps([{
                    "name": "book_ride",
                    "description": "Book a ride",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pickup_location": {"type": "string"},
                            "destination": {"type": "string"},
                            "ride_type": {"type": "string", "enum": ["plus", "comfort", "black", "pool", "xl"]}
                        },
                        "required": ["pickup_location", "destination", "ride_type"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Book me an Uber Premium from downtown to airport"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "book_ride",
                        "arguments": {
                            "pickup_location": "Downtown",
                            "destination": "Airport", 
                            "ride_type": "premium"  # Invalid enum value
                        }
                    }]
                }),
                "meta_source": "test_ride_booking",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": False
        },
        {
            "name": "Missing required field",
            "example": {
                "tools_json": json.dumps([{
                    "name": "book_ride",
                    "description": "Book a ride",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pickup_location": {"type": "string"},
                            "destination": {"type": "string"},
                            "ride_type": {"type": "string", "enum": ["plus", "comfort", "black", "pool", "xl"]}
                        },
                        "required": ["pickup_location", "destination", "ride_type"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Book me an Uber from downtown"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "book_ride", 
                        "arguments": {
                            "pickup_location": "Downtown",
                            "ride_type": "plus"
                            # Missing destination
                        }
                    }]
                }),
                "meta_source": "test_ride_booking",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": False
        }
    ],
    
    "weather_query": [
        {
            "name": "Valid weather query with proper location format",
            "example": {
                "tools_json": json.dumps([{
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "pattern": r"^.+,\s*.+$"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "What's the weather in Shanghai?"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "get_weather",
                        "arguments": {
                            "location": "Shanghai, China",
                            "units": "celsius"
                        }
                    }]
                }),
                "meta_source": "test_weather",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": True
        },
        {
            "name": "Invalid location format (missing comma)",
            "example": {
                "tools_json": json.dumps([{
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "pattern": r"^.+,\s*.+$"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "What's the weather in Shanghai?"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "get_weather",
                        "arguments": {
                            "location": "Shanghai China",  # Missing comma
                            "units": "celsius"
                        }
                    }]
                }),
                "meta_source": "test_weather",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": False
        },
        {
            "name": "Unicode location test",
            "example": {
                "tools_json": json.dumps([{
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "pattern": r"^.+,\s*.+$"},
                            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Weather in Tokyo please"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "get_weather",
                        "arguments": {
                            "location": "東京, 日本",  # Unicode location
                            "units": "celsius"
                        }
                    }]
                }),
                "meta_source": "test_weather",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": True
        }
    ],
    
    "timer_setting": [
        {
            "name": "Valid timer with integer seconds",
            "example": {
                "tools_json": json.dumps([{
                    "name": "set_timer",
                    "description": "Set a timer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                            "label": {"type": "string", "maxLength": 100}
                        },
                        "required": ["wait_seconds"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Set a 5 minute timer"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "set_timer",
                        "arguments": {
                            "wait_seconds": 300,
                            "label": "5 minute timer"
                        }
                    }]
                }),
                "meta_source": "test_timer",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": True
        },
        {
            "name": "Invalid timer with string seconds",
            "example": {
                "tools_json": json.dumps([{
                    "name": "set_timer",
                    "description": "Set a timer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                            "label": {"type": "string", "maxLength": 100}
                        },
                        "required": ["wait_seconds"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Set a timer for 5 minutes"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "set_timer",
                        "arguments": {
                            "wait_seconds": "300",  # String instead of int
                            "label": "5 minute timer"
                        }
                    }]
                }),
                "meta_source": "test_timer",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": False
        },
        {
            "name": "Timer with seconds out of range",
            "example": {
                "tools_json": json.dumps([{
                    "name": "set_timer",
                    "description": "Set a timer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                            "label": {"type": "string", "maxLength": 100}
                        },
                        "required": ["wait_seconds"]
                    }
                }]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Set a timer for 48 hours"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [{
                        "name": "set_timer",
                        "arguments": {
                            "wait_seconds": 172800,  # 48 hours > 24 hour max
                            "label": "48 hour timer"
                        }
                    }]
                }),
                "meta_source": "test_timer",
                "n_calls": 1,
                "difficulty": "simple",
                "valid": True
            },
            "should_pass": False
        }
    ],
    
    "parallel_calls": [
        {
            "name": "Valid parallel tool calls",
            "example": {
                "tools_json": json.dumps([
                    {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "pattern": r"^.+,\s*.+$"}
                            },
                            "required": ["location"]
                        }
                    },
                    {
                        "name": "set_timer",
                        "description": "Set a timer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 86400}
                            },
                            "required": ["wait_seconds"]
                        }
                    }
                ]),
                "messages_json": json.dumps([
                    {"role": "user", "content": "Check weather in Tokyo and set a 10 minute timer simultaneously"}
                ]),
                "target_json": json.dumps({
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"location": "Tokyo, Japan"}
                        },
                        {
                            "name": "set_timer", 
                            "arguments": {"wait_seconds": 600}
                        }
                    ]
                }),
                "meta_source": "test_parallel",
                "n_calls": 2,
                "difficulty": "parallel",
                "valid": True
            },
            "should_pass": True
        }
    ]
}


# Adversarial test cases with edge cases
ADVERSARIAL_TEST_CASES = [
    {
        "name": "Whitespace in string arguments",
        "example": {
            "tools_json": json.dumps([{
                "name": "search_restaurants",
                "description": "Search for restaurants",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "cuisine": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }]),
            "messages_json": json.dumps([
                {"role": "user", "content": "Find Italian restaurants near me"}
            ]),
            "target_json": json.dumps({
                "tool_calls": [{
                    "name": "search_restaurants",
                    "arguments": {
                        "location": "  San Francisco, CA  ",  # Extra whitespace
                        "cuisine": "  Italian  "
                    }
                }]
            }),
            "meta_source": "test_adversarial",
            "n_calls": 1,
            "difficulty": "simple",
            "valid": True
        },
        "should_pass": True  # Should handle whitespace gracefully
    },
    {
        "name": "Empty string in optional field",
        "example": {
            "tools_json": json.dumps([{
                "name": "book_appointment",
                "description": "Book an appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "notes": {"type": "string"}
                    },
                    "required": ["date"]
                }
            }]),
            "messages_json": json.dumps([
                {"role": "user", "content": "Book appointment for tomorrow"}
            ]),
            "target_json": json.dumps({
                "tool_calls": [{
                    "name": "book_appointment",
                    "arguments": {
                        "date": "2024-01-15",
                        "notes": ""  # Empty string for optional field
                    }
                }]
            }),
            "meta_source": "test_adversarial",
            "n_calls": 1,
            "difficulty": "simple",
            "valid": True
        },
        "should_pass": True
    },
    {
        "name": "Null value handling",
        "example": {
            "tools_json": json.dumps([{
                "name": "update_profile",
                "description": "Update user profile",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "name": {"type": "string"},
                        "bio": {"type": ["string", "null"]}  # Nullable field
                    },
                    "required": ["name"]
                }
            }]),
            "messages_json": json.dumps([
                {"role": "user", "content": "Update my name to John and clear my bio"}
            ]),
            "target_json": json.dumps({
                "tool_calls": [{
                    "name": "update_profile",
                    "arguments": {
                        "name": "John",
                        "bio": None  # Null value
                    }
                }]
            }),
            "meta_source": "test_adversarial",
            "n_calls": 1,
            "difficulty": "simple",
            "valid": True
        },
        "should_pass": True
    }
]


class TestQualityControl:
    """Test suite for quality control functions."""
    
    def test_validate_example_json_schema_valid(self):
        """Test validation of valid examples."""
        for family_name, test_cases in TOOL_FAMILY_TEST_CASES.items():
            for test_case in test_cases:
                if test_case["should_pass"]:
                    is_valid, errors = validate_example_json_schema(test_case["example"])
                    assert is_valid, f"Valid example {test_case['name']} failed validation: {errors}"
    
    def test_validate_example_json_schema_invalid(self):
        """Test validation correctly rejects invalid examples."""
        invalid_example = {
            "tools_json": "not valid json",
            "messages_json": "[]",
            "target_json": '{"tool_calls": []}',
            "meta_source": "test",
            "n_calls": 0,
            "difficulty": "simple",
            "valid": True
        }
        is_valid, errors = validate_example_json_schema(invalid_example)
        assert not is_valid
        assert len(errors) > 0
    
    def test_tool_call_schema_validation(self):
        """Test tool call schema validation."""
        # Test valid ride booking
        valid_call = {
            "name": "book_ride",
            "arguments": {
                "pickup_location": "Downtown",
                "destination": "Airport", 
                "ride_type": "plus"
            }
        }
        is_valid, error = validate_tool_call_schema(valid_call, "ride_booking")
        assert is_valid, f"Valid call failed: {error}"
        
        # Test invalid ride booking
        invalid_call = {
            "name": "book_ride",
            "arguments": {
                "pickup_location": "Downtown",
                "destination": "Airport",
                "ride_type": "premium"  # Invalid enum
            }
        }
        is_valid, error = validate_tool_call_schema(invalid_call, "ride_booking")
        assert not is_valid
        assert error is not None
    
    def test_toolbench_field_normalization(self):
        """Test ToolBench field normalization."""
        toolbench_call = {
            "name": "search_api",
            "arguments": {
                "city": "San Francisco",  # Should normalize to "location"
                "q": "restaurants",       # Should normalize to "search_query"
                "user_id": "123"         # Should normalize to "id"
            }
        }
        
        normalized = normalize_toolbench_fields(toolbench_call)
        expected_args = {
            "location": "San Francisco",
            "search_query": "restaurants", 
            "id": "123"
        }
        
        assert normalized["arguments"] == expected_args
    
    def test_schema_strict_example_generation(self):
        """Test generation of schema-strict examples."""
        example = generate_schema_strict_example("short_enums", {})
        assert example["valid"] is True
        assert example["n_calls"] == 1
        assert "schema_strict" in example["meta_source"]
        
        # Validate the generated example
        is_valid, errors = validate_example_json_schema(example)
        assert is_valid, f"Generated example failed validation: {errors}"
    
    def test_negative_example_generation(self):
        """Test generation of negative clarification examples."""
        template = CLARIFICATION_TEMPLATES[0]
        example = generate_negative_example(template)
        
        assert example["valid"] is True
        assert example["n_calls"] == 0
        assert example["meta_source"] == "negative_clarification"
        
        # Should have no tool calls
        target = json.loads(example["target_json"])
        assert len(target["tool_calls"]) == 0
        
        # Should have user query and assistant clarification
        messages = json.loads(example["messages_json"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "information" in messages[1]["content"].lower() or "details" in messages[1]["content"].lower()
    
    def test_no_tool_example_generation(self):
        """Test generation of no-tool explanation examples."""
        pattern = NO_TOOL_PATTERNS[0]
        example = generate_no_tool_example(pattern)
        
        assert example["valid"] is True
        assert example["n_calls"] == 0
        assert example["meta_source"] == "no_tool_explanation"
        
        # Should have no tool calls
        target = json.loads(example["target_json"])
        assert len(target["tool_calls"]) == 0
        
        # Should explain rather than call tools
        messages = json.loads(example["messages_json"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_adversarial_variants(self):
        """Test adversarial variant handling."""
        for test_case in ADVERSARIAL_TEST_CASES:
            is_valid, errors = validate_example_json_schema(test_case["example"])
            if test_case["should_pass"]:
                assert is_valid, f"Adversarial case {test_case['name']} should pass but failed: {errors}"
            else:
                assert not is_valid, f"Adversarial case {test_case['name']} should fail but passed"


def run_manual_test_suite() -> Dict[str, Any]:
    """Run the manual test suite and return results."""
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    # Test all tool families
    for family_name, test_cases in TOOL_FAMILY_TEST_CASES.items():
        for test_case in test_cases:
            results["total_tests"] += 1
            
            try:
                is_valid, errors = validate_example_json_schema(test_case["example"])
                
                if test_case["should_pass"] and is_valid:
                    results["passed"] += 1
                elif not test_case["should_pass"] and not is_valid:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "family": family_name,
                        "test": test_case["name"],
                        "expected": test_case["should_pass"],
                        "actual": is_valid,
                        "errors": errors
                    })
            except Exception as e:
                results["failed"] += 1
                results["failures"].append({
                    "family": family_name,
                    "test": test_case["name"],
                    "exception": str(e)
                })
    
    # Test adversarial cases
    for test_case in ADVERSARIAL_TEST_CASES:
        results["total_tests"] += 1
        
        try:
            is_valid, errors = validate_example_json_schema(test_case["example"])
            
            if test_case["should_pass"] and is_valid:
                results["passed"] += 1
            elif not test_case["should_pass"] and not is_valid:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "family": "adversarial",
                    "test": test_case["name"],
                    "expected": test_case["should_pass"],
                    "actual": is_valid,
                    "errors": errors
                })
        except Exception as e:
            results["failed"] += 1
            results["failures"].append({
                "family": "adversarial",
                "test": test_case["name"],
                "exception": str(e)
            })
    
    return results


if __name__ == "__main__":
    # Run manual test suite
    print("Running quality control test suite...")
    results = run_manual_test_suite()
    
    print("\nTest Results:")
    print(f"Total tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['passed']/results['total_tests']*100:.1f}%")
    
    if results['failures']:
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  {failure['family']}/{failure['test']}: {failure}")
