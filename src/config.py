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
CAP_DOLLY_NO_CALL = 7000      # No-call instruction following examples
CAP_WIKITEXT_NO_CALL = 5000   # No-call language understanding examples

GLOBAL_MAX_EXAMPLES = 100000

# Random seed for reproducibility
RANDOM_SEED = 42
