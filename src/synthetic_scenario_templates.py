"""
Enhanced scenario templates based on Berkeley Function-Calling Leaderboard patterns
"""

from typing import List, Dict, Any
import random

class EnhancedScenarioTemplates:
    """Enhanced scenario templates with more sophisticated patterns"""
    
    @staticmethod
    def get_function_templates() -> Dict[str, List[Dict[str, Any]]]:
        """Get function templates organized by category"""
        return {
            "mathematics": [
                {
                    "name": "calculate_area_triangle",
                    "description": "Calculates the area of a triangle given its base and height.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "base": {"type": "float", "description": "The base of the triangle."},
                            "height": {"type": "float", "description": "The height of the triangle."}
                        },
                        "required": ["base", "height"]
                    }
                },
                {
                    "name": "calculate_circle_area",
                    "description": "Calculates the area of a circle given its radius.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "radius": {"type": "float", "description": "The radius of the circle."}
                        },
                        "required": ["radius"]
                    }
                },
                {
                    "name": "calculate_rectangle_area",
                    "description": "Calculates the area of a rectangle given its width and height.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "width": {"type": "float", "description": "The width of the rectangle."},
                            "height": {"type": "float", "description": "The height of the rectangle."}
                        },
                        "required": ["width", "height"]
                    }
                },
                {
                    "name": "calculate_volume_sphere",
                    "description": "Calculates the volume of a sphere given its radius.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "radius": {"type": "float", "description": "The radius of the sphere."}
                        },
                        "required": ["radius"]
                    }
                }
            ],
            
            "statistics": [
                {
                    "name": "calculate_mean",
                    "description": "Calculates the mean of a list of numbers.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "numbers": {"type": "array", "items": {"type": "float"}, "description": "The list of numbers."}
                        },
                        "required": ["numbers"]
                    }
                },
                {
                    "name": "calculate_standard_deviation",
                    "description": "Calculates the standard deviation of a list of numbers.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "numbers": {"type": "array", "items": {"type": "float"}, "description": "The list of numbers."}
                        },
                        "required": ["numbers"]
                    }
                },
                {
                    "name": "calculate_median",
                    "description": "Calculates the median of a list of numbers.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "numbers": {"type": "array", "items": {"type": "float"}, "description": "The list of numbers."}
                        },
                        "required": ["numbers"]
                    }
                }
            ],
            
            "physics": [
                {
                    "name": "calculate_kinetic_energy",
                    "description": "Calculates the kinetic energy of an object.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "mass": {"type": "float", "description": "The mass of the object in kg."},
                            "velocity": {"type": "float", "description": "The velocity of the object in m/s."}
                        },
                        "required": ["mass", "velocity"]
                    }
                },
                {
                    "name": "calculate_gravitational_force",
                    "description": "Calculates gravitational force between two objects.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "mass1": {"type": "float", "description": "Mass of first object in kg."},
                            "mass2": {"type": "float", "description": "Mass of second object in kg."},
                            "distance": {"type": "float", "description": "Distance between objects in meters."}
                        },
                        "required": ["mass1", "mass2", "distance"]
                    }
                },
                {
                    "name": "calculate_acceleration",
                    "description": "Calculates acceleration given force and mass.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "force": {"type": "float", "description": "The applied force in Newtons."},
                            "mass": {"type": "float", "description": "The mass of the object in kg."}
                        },
                        "required": ["force", "mass"]
                    }
                }
            ],
            
            "finance": [
                {
                    "name": "calculate_compound_interest",
                    "description": "Calculates compound interest.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "principal": {"type": "float", "description": "Initial amount in dollars."},
                            "rate": {"type": "float", "description": "Annual interest rate (0-1)."},
                            "time": {"type": "integer", "description": "Time period in years."},
                            "frequency": {"type": "integer", "description": "Compounding frequency per year."}
                        },
                        "required": ["principal", "rate", "time", "frequency"]
                    }
                },
                {
                    "name": "calculate_loan_payment",
                    "description": "Calculates monthly loan payment.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "principal": {"type": "float", "description": "Loan amount in dollars."},
                            "rate": {"type": "float", "description": "Annual interest rate (0-1)."},
                            "months": {"type": "integer", "description": "Loan term in months."}
                        },
                        "required": ["principal", "rate", "months"]
                    }
                },
                {
                    "name": "calculate_present_value",
                    "description": "Calculates present value of future cash flow.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "future_value": {"type": "float", "description": "Future value in dollars."},
                            "rate": {"type": "float", "description": "Discount rate (0-1)."},
                            "periods": {"type": "integer", "description": "Number of periods."}
                        },
                        "required": ["future_value", "rate", "periods"]
                    }
                }
            ],
            
            "conversions": [
                {
                    "name": "convert_temperature",
                    "description": "Converts temperature between different scales.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "value": {"type": "float", "description": "Temperature value to convert."},
                            "from_scale": {"type": "string", "description": "Source temperature scale (C, F, K)."},
                            "to_scale": {"type": "string", "description": "Target temperature scale (C, F, K)."}
                        },
                        "required": ["value", "from_scale", "to_scale"]
                    }
                },
                {
                    "name": "convert_distance",
                    "description": "Converts distance between different units.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "value": {"type": "float", "description": "Distance value to convert."},
                            "from_unit": {"type": "string", "description": "Source distance unit."},
                            "to_unit": {"type": "string", "description": "Target distance unit."}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                },
                {
                    "name": "convert_weight",
                    "description": "Converts weight between different units.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "value": {"type": "float", "description": "Weight value to convert."},
                            "from_unit": {"type": "string", "description": "Source weight unit."},
                            "to_unit": {"type": "string", "description": "Target weight unit."}
                        },
                        "required": ["value", "from_unit", "to_unit"]
                    }
                }
            ],
            
            "data_analysis": [
                {
                    "name": "calculate_correlation",
                    "description": "Calculates correlation coefficient between two datasets.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "x": {"type": "array", "items": {"type": "float"}, "description": "First dataset."},
                            "y": {"type": "array", "items": {"type": "float"}, "description": "Second dataset."}
                        },
                        "required": ["x", "y"]
                    }
                },
                {
                    "name": "calculate_variance",
                    "description": "Calculates variance of a dataset.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "data": {"type": "array", "items": {"type": "float"}, "description": "The dataset."}
                        },
                        "required": ["data"]
                    }
                },
                {
                    "name": "find_outliers",
                    "description": "Identifies outliers in a dataset using IQR method.",
                    "parameters": {
                        "type": "dict",
                        "properties": {
                            "data": {"type": "array", "items": {"type": "float"}, "description": "The dataset."},
                            "threshold": {"type": "float", "description": "IQR multiplier threshold."}
                        },
                        "required": ["data", "threshold"]
                    }
                }
            ]
        }
    
    @staticmethod
    def get_scenario_prompts() -> Dict[str, List[str]]:
        """Get scenario prompt templates organized by category"""
        return {
            "mathematics": [
                "I'm working on a geometry project and need to calculate areas for several shapes: {shapes}. Could you help me with these calculations?",
                "For my engineering calculations, I need to determine the {measurement_type} of multiple objects: {objects}. Can you compute these for me?",
                "I'm designing {context} and need mathematical calculations for {items}. Please help with these computations.",
                "In my {field} work, I need to calculate {calculation_type} for several cases: {cases}. Could you process these?"
            ],
            
            "statistics": [
                "I'm analyzing {data_context} and need statistical measures for several datasets: {datasets}. Can you calculate {metric} for each?",
                "For my research on {topic}, I need to compute {statistical_measure} for multiple data groups: {groups}. Please help with the analysis.",
                "I'm studying {subject} and have collected data on {variables}. Could you calculate {measures} for each dataset?",
                "In my {analysis_type} analysis, I need {statistics} for the following data: {data_description}."
            ],
            
            "physics": [
                "I'm conducting physics experiments and need to calculate {physics_quantity} for several scenarios: {scenarios}. Can you help with these calculations?",
                "For my {experiment_type} experiment, I need to determine {physical_property} for multiple objects: {objects}. Please compute these values.",
                "I'm studying {physics_topic} and need calculations for {measurements} across different conditions: {conditions}.",
                "In my {physics_context}, I need to analyze {physics_parameters} for several cases: {cases}."
            ],
            
            "finance": [
                "I'm evaluating investment options and need to calculate {financial_metric} for several scenarios: {scenarios}. Could you help with these calculations?",
                "For my financial planning, I need to determine {calculation_type} for multiple {financial_instruments}: {details}. Please compute these.",
                "I'm analyzing {financial_context} and need calculations for {parameters} across different options: {options}.",
                "In my {investment_type} analysis, I need to evaluate {financial_measures} for several cases: {cases}."
            ],
            
            "conversions": [
                "I need to convert {measurement_type} for multiple values in my {context} project: {values}. Can you perform these conversions?",
                "For my international {purpose}, I need to convert {unit_type} between different systems: {conversion_details}. Please help with these.",
                "I'm working on {project_type} and need unit conversions for {items}: {conversion_list}. Could you handle these conversions?",
                "In my {application}, I need to convert {quantities} to different units for {reason}: {conversion_tasks}."
            ],
            
            "data_analysis": [
                "I'm performing data analysis on {dataset_context} and need to calculate {analysis_type} for several datasets: {datasets}. Can you help?",
                "For my {research_type} research, I need to analyze relationships between variables in multiple datasets: {dataset_descriptions}.",
                "I'm studying {subject} and need statistical analysis for {data_types}: {analysis_tasks}. Could you process these?",
                "In my {analytical_context}, I need to identify patterns and calculate metrics for {data_sources}: {specific_tasks}."
            ]
        }
    
    @staticmethod
    def get_realistic_contexts() -> Dict[str, List[str]]:
        """Get realistic contexts for different scenarios"""
        return {
            "mathematics": [
                "construction project", "architectural design", "engineering analysis", "manufacturing process",
                "scientific research", "academic assignment", "design validation", "quality control"
            ],
            "statistics": [
                "market research", "academic study", "clinical trial", "business analysis",
                "performance evaluation", "quality assessment", "survey analysis", "experimental data"
            ],
            "physics": [
                "laboratory experiment", "engineering simulation", "research project", "design optimization",
                "safety analysis", "performance testing", "scientific investigation", "material testing"
            ],
            "finance": [
                "investment portfolio", "loan comparison", "retirement planning", "business evaluation",
                "risk assessment", "financial modeling", "budget analysis", "investment strategy"
            ],
            "conversions": [
                "international trade", "scientific research", "engineering project", "travel planning",
                "recipe adaptation", "manufacturing process", "academic work", "technical documentation"
            ],
            "data_analysis": [
                "business intelligence", "scientific research", "market analysis", "performance metrics",
                "quality control", "customer behavior", "operational efficiency", "trend analysis"
            ]
        }
    
    @staticmethod
    def generate_realistic_data(category: str, function_name: str) -> Dict[str, Any]:
        """Generate realistic parameter data for a given function"""
        
        data_generators = {
            "calculate_area_triangle": lambda: {
                "base": round(random.uniform(5, 50), 1),
                "height": round(random.uniform(3, 40), 1)
            },
            "calculate_circle_area": lambda: {
                "radius": round(random.uniform(2, 25), 1)
            },
            "calculate_rectangle_area": lambda: {
                "width": round(random.uniform(5, 50), 1),
                "height": round(random.uniform(3, 40), 1)
            },
            "calculate_volume_sphere": lambda: {
                "radius": round(random.uniform(1, 20), 1)
            },
            "calculate_mean": lambda: {
                "numbers": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))]
            },
            "calculate_standard_deviation": lambda: {
                "numbers": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))]
            },
            "calculate_median": lambda: {
                "numbers": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))]
            },
            "calculate_kinetic_energy": lambda: {
                "mass": round(random.uniform(0.5, 100), 1),
                "velocity": round(random.uniform(1, 50), 1)
            },
            "calculate_gravitational_force": lambda: {
                "mass1": round(random.uniform(1, 1000), 1),
                "mass2": round(random.uniform(1, 1000), 1),
                "distance": round(random.uniform(0.1, 100), 2)
            },
            "calculate_acceleration": lambda: {
                "force": round(random.uniform(1, 1000), 1),
                "mass": round(random.uniform(0.5, 100), 1)
            },
            "calculate_compound_interest": lambda: {
                "principal": random.randint(1000, 100000),
                "rate": round(random.uniform(0.01, 0.15), 3),
                "time": random.randint(1, 30),
                "frequency": random.choice([1, 2, 4, 12])
            },
            "calculate_loan_payment": lambda: {
                "principal": random.randint(10000, 500000),
                "rate": round(random.uniform(0.02, 0.12), 3),
                "months": random.choice([12, 24, 36, 60, 120, 180, 240, 360])
            },
            "calculate_present_value": lambda: {
                "future_value": random.randint(1000, 100000),
                "rate": round(random.uniform(0.01, 0.15), 3),
                "periods": random.randint(1, 20)
            },
            "convert_temperature": lambda: {
                "value": round(random.uniform(-20, 100), 1),
                "from_scale": random.choice(["C", "F", "K"]),
                "to_scale": random.choice(["C", "F", "K"])
            },
            "convert_distance": lambda: {
                "value": round(random.uniform(1, 1000), 2),
                "from_unit": random.choice(["meters", "feet", "inches", "km", "miles"]),
                "to_unit": random.choice(["meters", "feet", "inches", "km", "miles"])
            },
            "convert_weight": lambda: {
                "value": round(random.uniform(1, 1000), 2),
                "from_unit": random.choice(["kg", "lbs", "grams", "ounces"]),
                "to_unit": random.choice(["kg", "lbs", "grams", "ounces"])
            },
            "calculate_correlation": lambda: {
                "x": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))],
                "y": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))]
            },
            "calculate_variance": lambda: {
                "data": [round(random.uniform(1, 100), 1) for _ in range(random.randint(5, 10))]
            },
            "find_outliers": lambda: {
                "data": [round(random.uniform(1, 100), 1) for _ in range(random.randint(8, 15))],
                "threshold": round(random.uniform(1.0, 3.0), 1)
            }
        }
        
        generator = data_generators.get(function_name)
        return generator() if generator else {}
    
    @staticmethod
    def format_function_call(function_name: str, parameters: Dict[str, Any]) -> str:
        """Format a function call string from parameters"""
        param_strings = []
        for key, value in parameters.items():
            if isinstance(value, str):
                param_strings.append(f"{key}='{value}'")
            elif isinstance(value, list):
                param_strings.append(f"{key}={value}")
            else:
                param_strings.append(f"{key}={value}")
        
        return f"{function_name}({', '.join(param_strings)})"
