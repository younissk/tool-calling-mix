"""Configuration constants for the tool-calling-sft-mix project."""


# File paths
OPENFUNCTIONS_FILES = [
    "data/gorilla_openfunctions_v1_train.json",
    "data/gorilla_openfunctions_v1_test.json",
]

TOOLBENCH_FILES = [
    "data/ToolBench/toolllama_G123_dfs_train.json",
    "data/ToolBench/toolllama_G123_dfs_eval.json",
]

# Feature flags
USE_XLAM = True
USE_TOOLBENCH = True

# Dataset caps - optimized for multi-call and no-call balance
CAP_XLAM = 20000              # Single and multi-call tool examples
CAP_OPENFUNCTIONS = 15000     # Tool calling examples  
CAP_TOOLBENCH = 50000         # Multi-call rich examples (~48% multi-call)
CAP_DOLLY_NO_CALL = 10000      # No-call instruction following examples
CAP_WIKITEXT_NO_CALL = 7000   # No-call language understanding examples

GLOBAL_MAX_EXAMPLES = 100000

# Quality control settings
ENABLE_QUALITY_CONTROLS = True
ENABLE_SCHEMA_STRICT_EXEMPLARS = True
ENABLE_NEGATIVE_EXAMPLES = True
ENABLE_BALANCED_NO_TOOL = True
ENABLE_TOOLBENCH_NORMALIZATION = True
ENABLE_ADVERSARIAL_VARIANTS = True

# Quality exemplar counts
CAP_QUALITY_EXEMPLARS = 1000      # Schema-strict examples for failure patterns
CAP_NEGATIVE_EXAMPLES = 500       # Clarification request examples  
CAP_BALANCED_NO_TOOL = 800        # Tool-triggering explanations (no calls)

# Quality control thresholds
MIN_NO_CALL_PERCENTAGE = 20       # Minimum percentage of no-call examples
MAX_NO_CALL_PERCENTAGE = 25       # Maximum percentage of no-call examples
ADVERSARIAL_VARIANT_RATIO = 0.1   # 10% of examples get adversarial variants

# Random seed for reproducibility
RANDOM_SEED = 42
