"""Quality control utilities for generating high-quality tool calling examples."""

import json
import random
from typing import Dict, Any, List, Optional, Tuple
import jsonschema


# Schema-strict exemplars for common failure patterns
SCHEMA_STRICT_EXEMPLARS = {
    "short_enums": [
        {
            "function_name": "book_ride",
            "parameter": "ride_type",
            "valid_values": ["plus", "comfort", "black", "pool", "xl"],
            "correct_examples": [
                {"ride_type": "plus"}, 
                {"ride_type": "comfort"},
                {"ride_type": "black"}
            ],
            "incorrect_examples": [
                {"ride_type": "Uber Plus"},  # Too verbose
                {"ride_type": "premium"},    # Wrong value
                {"ride_type": "COMFORT"}     # Wrong case
            ]
        },
        {
            "function_name": "set_temperature",
            "parameter": "mode",
            "valid_values": ["heat", "cool", "auto", "off"],
            "correct_examples": [
                {"mode": "heat"}, 
                {"mode": "cool"},
                {"mode": "auto"}
            ],
            "incorrect_examples": [
                {"mode": "heating"},     # Too verbose
                {"mode": "Hot"},         # Wrong case
                {"mode": "automatic"}    # Wrong value
            ]
        }
    ],
    
    "state_abbreviations": [
        {
            "function_name": "search_restaurants",
            "parameter": "state",
            "valid_values": ["CA", "NY", "TX", "FL", "WA", "IL", "PA", "OH"],
            "correct_examples": [
                {"state": "CA"}, 
                {"state": "NY"},
                {"state": "TX"}
            ],
            "incorrect_examples": [
                {"state": "California"},  # Full name
                {"state": "ca"},          # Wrong case
                {"state": "CALIF"}        # Wrong abbreviation
            ]
        }
    ],
    
    "exact_locales": [
        {
            "function_name": "get_weather",
            "parameter": "location",
            "valid_values": [
                "Shanghai, China", "上海, 中国", 
                "New York, NY", "Paris, France", 
                "Tokyo, Japan", "東京, 日本"
            ],
            "correct_examples": [
                {"location": "Shanghai, China"}, 
                {"location": "上海, 中国"},
                {"location": "New York, NY"}
            ],
            "incorrect_examples": [
                {"location": "Shanghai"},         # Missing country
                {"location": "Shanghai China"},   # Missing comma
                {"location": "shanghai, china"}   # Wrong case
            ]
        }
    ],
    
    "type_correct_ints": [
        {
            "function_name": "set_timer",
            "parameter": "wait_seconds",
            "data_type": "integer",
            "correct_examples": [
                {"wait_seconds": 30}, 
                {"wait_seconds": 300},
                {"wait_seconds": 0}
            ],
            "incorrect_examples": [
                {"wait_seconds": "30"},      # String instead of int
                {"wait_seconds": 30.0},      # Float instead of int
                {"wait_seconds": "5 minutes"} # Human readable string
            ]
        },
        {
            "function_name": "book_appointment",
            "parameter": "duration_minutes",
            "data_type": "integer",
            "correct_examples": [
                {"duration_minutes": 60}, 
                {"duration_minutes": 90},
                {"duration_minutes": 30}
            ],
            "incorrect_examples": [
                {"duration_minutes": "1 hour"},  # Human readable
                {"duration_minutes": 60.5},      # Float
                {"duration_minutes": "60"}       # String
            ]
        }
    ]
}


# Negative/clarification example templates
CLARIFICATION_TEMPLATES = [
    {
        "user_query": "Book me an Uber ride",
        "missing_fields": ["pickup_location", "destination", "ride_type"],
        "assistant_response": "I'd be happy to help you book an Uber ride! However, I need some additional information to proceed:\n\n- What's your pickup location?\n- Where would you like to go?\n- What type of ride would you prefer? (options: plus, comfort, black, pool, xl)\n\nOnce you provide these details, I can book the ride for you."
    },
    {
        "user_query": "Set a timer",
        "missing_fields": ["duration"],
        "assistant_response": "I can help you set a timer! How long would you like the timer to run? Please specify the duration in seconds (e.g., 300 for 5 minutes, 1800 for 30 minutes)."
    },
    {
        "user_query": "Check the weather",
        "missing_fields": ["location"],
        "assistant_response": "I can check the weather for you! Which location would you like the weather information for? Please provide the city and country/state in the format 'City, Country' (e.g., 'New York, NY' or 'Shanghai, China')."
    },
    {
        "user_query": "Book a restaurant reservation",
        "missing_fields": ["restaurant_name", "date", "time", "party_size"],
        "assistant_response": "I'd be happy to help you make a restaurant reservation! I need the following information:\n\n- Restaurant name\n- Date (YYYY-MM-DD format)\n- Time (HH:MM format)\n- Number of people in your party\n\nCould you please provide these details?"
    }
]


# ToolBench field normalization mappings
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
    "appointment_time": "time",
    
    # Common parameter aliases
    "query": "search_query",
    "q": "search_query",
    "search_term": "search_query",
    "keywords": "search_query",
    
    # ID aliases
    "identifier": "id",
    "user_id": "id", 
    "item_id": "id",
    "reference": "id"
}


# JSON Schema validators for common tool patterns
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
    },
    
    "weather_query": {
        "type": "object", 
        "properties": {
            "location": {"type": "string", "pattern": r"^.+,\s*.+$"},  # Require comma-separated format
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"],
        "additionalProperties": False
    },
    
    "timer_setting": {
        "type": "object",
        "properties": {
            "wait_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},  # 1 sec to 24 hours
            "label": {"type": "string", "maxLength": 100}
        },
        "required": ["wait_seconds"],
        "additionalProperties": False
    }
}


# No-tool triggering patterns that should NOT result in tool calls
NO_TOOL_PATTERNS = [
    {
        "query": "Explain what an Uber Comfort is",
        "response": "Uber Comfort is a premium ride option that offers more legroom and highly-rated drivers. It typically costs slightly more than UberX but provides additional comfort features like temperature control requests and top-rated drivers with newer vehicles."
    },
    {
        "query": "What's the difference between weather and climate?",
        "response": "Weather refers to short-term atmospheric conditions (like today's temperature and precipitation), while climate describes long-term weather patterns averaged over decades. Weather can change daily, but climate represents the typical conditions for a region over many years."
    },
    {
        "query": "How do restaurant reservations work?",
        "response": "Restaurant reservations typically involve contacting the restaurant (by phone, website, or app) to secure a table for a specific date, time, and party size. Many restaurants now use online booking systems, and some require deposits for larger groups or special occasions."
    },
    {
        "query": "Tell me about timer apps on smartphones",
        "response": "Smartphone timer apps allow users to set countdown timers for various activities like cooking, workouts, or time management. Most phones have built-in timer functionality that can run in the background, send notifications when time expires, and often include features like multiple simultaneous timers and custom labels."
    },
    {
        "query": "What are the benefits of using APIs for travel booking?",
        "response": "API-based travel booking offers several advantages: real-time availability and pricing, automated comparison across multiple providers, seamless integration with other services, reduced manual input errors, and the ability to handle complex multi-step bookings programmatically."
    }
]


def normalize_toolbench_fields(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ToolBench field names to canonical versions."""
    if not isinstance(tool_call, dict) or "arguments" not in tool_call:
        return tool_call
    
    normalized_args = {}
    for key, value in tool_call["arguments"].items():
        canonical_key = TOOLBENCH_FIELD_MAPPINGS.get(key, key)
        normalized_args[canonical_key] = value
    
    return {
        **tool_call,
        "arguments": normalized_args
    }


def validate_tool_call_schema(tool_call: Dict[str, Any], function_name: str) -> Tuple[bool, Optional[str]]:
    """Validate tool call arguments against known schemas."""
    if function_name not in TOOL_SCHEMAS:
        return True, None  # No schema to validate against
    
    try:
        schema = TOOL_SCHEMAS[function_name]
        jsonschema.validate(tool_call.get("arguments", {}), schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)


def generate_schema_strict_example(pattern_type: str, function_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a schema-strict example for a given pattern type."""
    if pattern_type not in SCHEMA_STRICT_EXEMPLARS:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    patterns = SCHEMA_STRICT_EXEMPLARS[pattern_type]
    pattern = random.choice(patterns)
    
    if "correct_examples" not in pattern:
        return {}
    
    correct_example = random.choice(pattern["correct_examples"])
    
    # Create tool specification
    tools = [{
        "name": pattern["function_name"],
        "description": f"Function demonstrating {pattern_type} pattern",
        "parameters": {
            "type": "object",
            "properties": {
                param: {"type": "string" if "type" not in pattern else pattern.get("data_type", "string")}
                for param in correct_example.keys()
            },
            "required": list(correct_example.keys())
        }
    }]
    
    # Create messages
    messages = [{
        "role": "user",
        "content": f"Please call {pattern['function_name']} with appropriate parameters"
    }]
    
    # Create target
    target = {
        "tool_calls": [{
            "name": pattern["function_name"],
            "arguments": correct_example
        }]
    }
    
    return {
        "tools_json": json.dumps(tools),
        "messages_json": json.dumps(messages),
        "target_json": json.dumps(target),
        "meta_source": f"schema_strict_{pattern_type}",
        "n_calls": 1,
        "difficulty": "simple",
        "valid": True
    }


def generate_negative_example(template: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a negative example showing clarification request."""
    messages = [
        {"role": "user", "content": template["user_query"]},
        {"role": "assistant", "content": template["assistant_response"]}
    ]
    
    return {
        "tools_json": "[]",  # No tools needed for clarification
        "messages_json": json.dumps(messages),
        "target_json": json.dumps({"tool_calls": []}),  # No tool calls
        "meta_source": "negative_clarification",
        "n_calls": 0,
        "difficulty": "simple", 
        "valid": True
    }


def generate_no_tool_example(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a no-tool example that looks tool-triggering but isn't."""
    messages = [
        {"role": "user", "content": pattern["query"]},
        {"role": "assistant", "content": pattern["response"]}
    ]
    
    return {
        "tools_json": "[]",
        "messages_json": json.dumps(messages),
        "target_json": json.dumps({"tool_calls": []}),
        "meta_source": "no_tool_explanation",
        "n_calls": 0,
        "difficulty": "simple",
        "valid": True
    }


def create_adversarial_variant(example: Dict[str, Any]) -> Dict[str, Any]:
    """Create adversarial variants with edge cases."""
    try:
        messages = json.loads(example["messages_json"])
        target = json.loads(example["target_json"])
        
        # Apply random adversarial transformations
        adversarial_type = random.choice([
            "whitespace", "punctuation", "unicode", "falsy_optional"
        ])
        
        if adversarial_type == "whitespace" and target["tool_calls"]:
            # Add extra whitespace to string arguments
            for call in target["tool_calls"]:
                for key, value in call["arguments"].items():
                    if isinstance(value, str):
                        call["arguments"][key] = f"  {value}  "
        
        elif adversarial_type == "punctuation" and messages:
            # Add punctuation variations to user message
            user_msg = next((msg for msg in messages if msg["role"] == "user"), None)
            if user_msg:
                content = user_msg["content"]
                # Add random punctuation
                variations = [f"{content}!", f"{content}?", f"{content}...", f"{content}."]
                user_msg["content"] = random.choice(variations)
        
        elif adversarial_type == "unicode" and target["tool_calls"]:
            # Test unicode in location fields
            for call in target["tool_calls"]:
                for key, value in call["arguments"].items():
                    if isinstance(value, str) and ("location" in key.lower() or "city" in key.lower()):
                        # Add unicode variants like "东京, 日本" for Tokyo
                        unicode_variants = {
                            "Tokyo": "東京",
                            "Shanghai": "上海", 
                            "Beijing": "北京",
                            "Seoul": "서울"
                        }
                        for eng, unicode_val in unicode_variants.items():
                            if eng in value:
                                call["arguments"][key] = value.replace(eng, unicode_val)
        
        elif adversarial_type == "falsy_optional" and target["tool_calls"]:
            # Test falsy values for optional parameters
            for call in target["tool_calls"]:
                # Add an optional parameter with falsy value
                falsy_values = ["", "null", None, 0, False]
                call["arguments"]["optional_param"] = random.choice(falsy_values)
        
        return {
            **example,
            "messages_json": json.dumps(messages),
            "target_json": json.dumps(target),
            "meta_source": f"{example['meta_source']}_adversarial_{adversarial_type}",
        }
        
    except Exception:
        # Return original if adversarial generation fails
        return example


def validate_example_json_schema(example: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that an example's JSON fields are valid and well-formed."""
    errors = []
    
    try:
        # Validate tools_json
        tools = json.loads(example["tools_json"])
        if not isinstance(tools, list):
            errors.append("tools_json must be a list")
        else:
            for i, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    errors.append(f"Tool {i} must be a dict")
                elif "name" not in tool:
                    errors.append(f"Tool {i} missing required 'name' field")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in tools_json: {e}")
    
    try:
        # Validate messages_json
        messages = json.loads(example["messages_json"])
        if not isinstance(messages, list):
            errors.append("messages_json must be a list")
        else:
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"Message {i} must be a dict")
                elif "role" not in msg or "content" not in msg:
                    errors.append(f"Message {i} missing 'role' or 'content'")
                elif msg["role"] not in ["user", "assistant", "tool", "system"]:
                    errors.append(f"Message {i} has invalid role: {msg['role']}")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in messages_json: {e}")
    
    try:
        # Validate target_json
        target = json.loads(example["target_json"])
        if not isinstance(target, dict):
            errors.append("target_json must be a dict")
        elif "tool_calls" not in target:
            errors.append("target_json missing 'tool_calls' field")
        elif not isinstance(target["tool_calls"], list):
            errors.append("target_json.tool_calls must be a list")
        else:
            for i, call in enumerate(target["tool_calls"]):
                if not isinstance(call, dict):
                    errors.append(f"Tool call {i} must be a dict")
                elif "name" not in call or "arguments" not in call:
                    errors.append(f"Tool call {i} missing 'name' or 'arguments'")
                elif not isinstance(call["arguments"], dict):
                    errors.append(f"Tool call {i} arguments must be a dict")
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in target_json: {e}")
    
    return len(errors) == 0, errors


def filter_quality_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter examples to only include high-quality ones that pass basic validation."""
    filtered = []
    
    for example in examples:
        try:
            # Parse JSON fields to ensure they're valid
            target = json.loads(example["target_json"])
            messages = json.loads(example["messages_json"])
            tools = json.loads(example["tools_json"])
            
            # Basic structure validation
            if not isinstance(messages, list) or not isinstance(tools, list):
                continue
                
            # Ensure at least one user message
            if not any(msg.get("role") == "user" for msg in messages):
                continue
                
            # Update n_calls if needed instead of filtering
            actual_n_calls = len(target.get("tool_calls", []))
            if actual_n_calls != example.get("n_calls", 0):
                example["n_calls"] = actual_n_calls
                
            filtered.append(example)
                
        except Exception:
            # Skip examples that fail parsing
            continue
    
    return filtered
