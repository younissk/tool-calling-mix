#!/usr/bin/env python3
"""
Synthetic Parallel Tool Call Dataset Generator with Agent Mode

This module generates synthetic parallel tool call examples using the Falcon-h1-34B model
via llama.cpp endpoint. It creates scenarios where multiple different tools are called in parallel.
"""

import random
import requests
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm
from src.synthetic_scenario_templates import EnhancedScenarioTemplates

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParallelScenario:
    """Represents a parallel function calling scenario template"""
    category: str
    user_prompt_template: str
    functions: List[Dict[str, Any]]
    ground_truth_template: List[str]
    execution_result_type: str = "exact_match"

class SyntheticParallelGenerator:
    """Generator for synthetic parallel tool call data with agent mode"""
    
    def __init__(self, hf_url: Optional[str] = None, model_name: str = "falcon-h1-34b"):
        """
        Initialize the generator
        
        Args:
            hf_url: The URL of your Falcon-h1-34B llama.cpp instance
            model_name: Name of the model for identification
        """
        self.hf_url = hf_url or os.getenv("HF_URL")
        self.model_name = model_name
        self.generated_count = 0
        
        # Initialize scenario templates
        self.scenarios = self._create_scenario_templates()
        
        # Create output directory
        os.makedirs("data", exist_ok=True)
        
    def _create_scenario_templates(self) -> List[ParallelScenario]:
        """Create templates for different types of parallel scenarios using enhanced templates"""
        
        function_templates = EnhancedScenarioTemplates.get_function_templates()
        prompt_templates = EnhancedScenarioTemplates.get_scenario_prompts()
        
        scenarios = []
        
        # Create cross-category scenarios for true parallel tool calling
        categories = list(function_templates.keys())
        for i, primary_category in enumerate(categories):
            # Get 2-3 other random categories for mixing
            other_categories = categories[:i] + categories[i+1:]
            num_extra = random.randint(1, 2)  # 2-3 tools total
            mix_categories = random.sample(other_categories, num_extra)
            mix_categories.append(primary_category)
            
            # Create scenario with functions from multiple categories
            for prompt_template in prompt_templates.get(primary_category, []):
                selected_functions = []
                for cat in mix_categories:
                    if cat in function_templates:
                        # Pick 1 random function from each category
                        func = random.choice(function_templates[cat])
                        selected_functions.append(func)
                
                if selected_functions:
                    scenario = ParallelScenario(
                        category=f"mixed_{primary_category}",
                        user_prompt_template=prompt_template,
                        functions=selected_functions,
                        ground_truth_template=[],  # Will be generated dynamically
                        execution_result_type="exact_match"
                    )
                    scenarios.append(scenario)
        
        return scenarios
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Call the Falcon-h1-34B model via llama.cpp endpoint"""
        if not self.hf_url:
            logger.warning("No HF_URL provided, skipping LLM generation")
            return None
            
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["\n\n", "###", "```"]
            }
            
            response = requests.post(
                f"{self.hf_url}/v1/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            return None
    
    def _generate_parallel_scenario_data(self, scenario: ParallelScenario) -> Dict[str, Any]:
        """Generate data for a parallel scenario using multiple different tools"""
        
        # Try LLM generation first if available
        if self.hf_url:
            # Create prompt that emphasizes using different tools
            function_names = [f["name"] for f in scenario.functions]
            function_descs = [f"{f['name']}: {f['description']}" for f in scenario.functions]
            
            generation_prompt = f"""Generate a realistic scenario where multiple different tools need to be used in parallel.

Available Tools:
{chr(10).join(function_descs)}

Create a natural user request that requires using ALL of these tools ({len(function_names)} tools) together.
The request should make sense to use these tools in parallel (at the same time).

Example format:
User: "I need to analyze temperature data: calculate the mean of [23.5, 25.1, 22.8], convert 72°F to Celsius, and find outliers in [19.5, 45.2, 22.1, 23.4] with threshold 2.0"
Tool Calls:
calculate_mean(numbers=[23.5, 25.1, 22.8])
convert_temperature(value=72, from_scale='F', to_scale='C')
find_outliers(data=[19.5, 45.2, 22.1, 23.4], threshold=2.0)

Your response (create a realistic scenario using these specific tools: {', '.join(function_names)}):"""

            response = self._call_llm(generation_prompt, max_tokens=400)
            if response:
                try:
                    return self._parse_llm_response(response, scenario)
                except Exception as e:
                    logger.warning(f"Failed to parse LLM response: {e}. Using enhanced templates.")
        
        # Use enhanced templates as sophisticated fallback
        try:
            return self._generate_enhanced_scenario(scenario)
        except Exception as e:
            logger.warning(f"Enhanced generation failed: {e}. Using basic fallback.")
            
        return self._generate_fallback_scenario(scenario)
    
    def _generate_enhanced_scenario(self, scenario: ParallelScenario) -> Dict[str, Any]:
        """Generate enhanced scenario using multiple different tools"""
        
        # Generate one parameter set for each different function
        tool_calls = []
        for function in scenario.functions:
            function_name = function["name"]
            params = EnhancedScenarioTemplates.generate_realistic_data(scenario.category, function_name)
            if params:
                call_str = EnhancedScenarioTemplates.format_function_call(function_name, params)
                tool_calls.append(call_str)
        
        # Generate realistic user request that combines all tools
        contexts = EnhancedScenarioTemplates.get_realistic_contexts()
        context_options = contexts.get(scenario.category, ["my project"])
        context = random.choice(context_options)
        
        # Create natural language description based on all functions
        user_request = self._generate_natural_request(scenario, tool_calls, context)
        
        return {
            "id": f"synthetic_parallel_{self.generated_count}",
            "question": [[{"role": "user", "content": user_request}]],
            "function": scenario.functions,  # Include all available functions
            "execution_result_type": ["exact_match"] * len(tool_calls),
            "ground_truth": tool_calls
        }
    
    def _generate_natural_request(self, scenario: ParallelScenario, tool_calls: List[str], context: str) -> str:
        """Generate natural language request that requires multiple different tools"""
        
        # Extract function names and parameters for better prompt construction
        function_tasks = []
        for call in tool_calls:
            name = call.split("(")[0]
            # Convert function_name to readable form
            readable_name = name.replace("_", " ")
            function_tasks.append(readable_name)
        
        # Create a natural request combining all tasks
        tasks_str = ", ".join(function_tasks[:-1])
        if len(function_tasks) > 1:
            tasks_str += f" and {function_tasks[-1]}"
        else:
            tasks_str = function_tasks[0]
            
        return f"For {context}, I need to {tasks_str}. Can you help with these calculations?"
    
    def _parse_llm_response(self, response: str, scenario: ParallelScenario) -> Dict[str, Any]:
        """Parse LLM response and create a structured scenario"""
        lines = response.strip().split('\n')
        
        # Extract user request and tool calls
        user_request = ""
        tool_calls = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("User:"):
                user_request = line.split(":", 1)[1].strip().strip('"')
            elif any(f"{func['name']}(" in line for func in scenario.functions):
                # Found a tool call
                tool_calls.append(line)
        
        if not user_request:
            user_request = f"I need to perform multiple calculations for {scenario.category}."
        
        if not tool_calls:
            # Generate fallback tool calls
            tool_calls = []
            for function in scenario.functions:
                params = EnhancedScenarioTemplates.generate_realistic_data(scenario.category, function["name"])
                if params:
                    call_str = EnhancedScenarioTemplates.format_function_call(function["name"], params)
                    tool_calls.append(call_str)
        
        return {
            "id": f"synthetic_parallel_{self.generated_count}",
            "question": [[{"role": "user", "content": user_request}]],
            "function": scenario.functions,
            "execution_result_type": ["exact_match"] * len(tool_calls),
            "ground_truth": tool_calls
        }
    
    def _generate_fallback_scenario(self, scenario: ParallelScenario) -> Dict[str, Any]:
        """Generate a basic fallback scenario with multiple tools"""
        
        tool_calls = []
        for function in scenario.functions:
            function_name = function["name"]
            # Generate simple parameters
            if "numbers" in str(function):
                tool_calls.append(f"{function_name}(numbers=[1.0, 2.0, 3.0])")
            elif "radius" in str(function):
                tool_calls.append(f"{function_name}(radius=5.0)")
            elif "temperature" in str(function):
                tool_calls.append(f"{function_name}(value=20, from_scale='C', to_scale='F')")
            else:
                tool_calls.append(f"{function_name}(value=1)")
        
        return {
            "id": f"synthetic_parallel_{self.generated_count}",
            "question": [[{"role": "user", "content": f"I need to perform multiple calculations for {scenario.category}."}]],
            "function": scenario.functions,
            "execution_result_type": ["exact_match"] * len(tool_calls),
            "ground_truth": tool_calls
        }

    def generate_synthetic_examples(self, target_count: int = 1000) -> List[Dict[str, Any]]:
        """Generate a set of synthetic parallel scenarios with real-time saving"""
        
        synthetic_data = []
        scenarios_per_category = target_count // len(self.scenarios)
        
        logger.info(f"Generating {target_count} synthetic examples across {len(self.scenarios)} categories")
        
        # Create progress bar
        pbar = tqdm(total=target_count, desc="Generating examples")
        
        # Track last save time
        last_save_time = time.time()
        save_interval = 60  # Save every 60 seconds
        
        for scenario in self.scenarios:
            logger.info(f"Generating {scenarios_per_category} examples for category: {scenario.category}")
            
            for i in range(scenarios_per_category):
                try:
                    instance = self._generate_parallel_scenario_data(scenario)
                    synthetic_data.append(instance)
                    self.generated_count += 1
                    pbar.update(1)
                    
                    # Save progress periodically
                    current_time = time.time()
                    if current_time - last_save_time >= save_interval:
                        self._save_progress(synthetic_data)
                        last_save_time = current_time
                    
                    # Small delay to avoid overwhelming the API
                    if self.hf_url:
                        time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error generating scenario {i} for category {scenario.category}: {e}")
                    continue
        
        # Fill remaining with random categories if needed
        while len(synthetic_data) < target_count:
            scenario = random.choice(self.scenarios)
            try:
                instance = self._generate_parallel_scenario_data(scenario)
                synthetic_data.append(instance)
                self.generated_count += 1
                pbar.update(1)
                
                # Save progress periodically
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self._save_progress(synthetic_data)
                    last_save_time = current_time
                    
            except Exception as e:
                logger.error(f"Error generating additional scenario: {e}")
                break
        
        pbar.close()
        
        # Final save
        self._save_progress(synthetic_data, is_final=True)
        
        return synthetic_data
    
    def _save_progress(self, data: List[Dict[str, Any]], is_final: bool = False) -> None:
        """Save current progress to file"""
        try:
            filename = "synthetic_parallel_tool_calls.json" if is_final else "synthetic_parallel_tool_calls_partial.json"
            filepath = os.path.join("data", filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if not is_final:
                logger.info(f"Progress saved: {len(data)} examples generated")
            else:
                logger.info(f"Final dataset saved: {len(data)} examples")
                
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean the generated data"""
        validated_data = []
        
        for item in data:
            try:
                # Check required fields
                required_fields = ["id", "question", "function", "execution_result_type", "ground_truth"]
                if not all(field in item for field in required_fields):
                    logger.warning(f"Skipping item {item.get('id', 'unknown')} - missing required fields")
                    continue
                
                # Validate structure
                if not isinstance(item["question"], list) or not item["question"]:
                    logger.warning(f"Skipping item {item['id']} - invalid question format")
                    continue
                
                if not isinstance(item["function"], list) or not item["function"]:
                    logger.warning(f"Skipping item {item['id']} - invalid function format")
                    continue
                
                if not isinstance(item["ground_truth"], list) or not item["ground_truth"]:
                    logger.warning(f"Skipping item {item['id']} - invalid ground_truth format")
                    continue
                
                # Ensure multiple different tools are used
                tool_names = set(call.split("(")[0] for call in item["ground_truth"])
                if len(tool_names) < 2:
                    logger.warning(f"Skipping item {item['id']} - not enough different tools used")
                    continue
                
                validated_data.append(item)
                
            except Exception as e:
                logger.error(f"Error validating item: {e}")
                continue
        
        logger.info(f"Validated {len(validated_data)}/{len(data)} examples")
        return validated_data