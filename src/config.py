"""Configuration constants for the tool-calling-sft-mix project."""


# File paths
OPENFUNCTIONS_FILES = [
    "data/gorilla_openfunctions_v1_train.json",
    "data/gorilla_openfunctions_v1_test.json",
]

# Feature flags
USE_XLAM = True

# Dataset caps
CAP_XLAM = 20000
CAP_OPENFUNCTIONS = 15000
CAP_DOLLY_NO_CALL = 5000
CAP_WIKITEXT_NO_CALL = 3000

# Global limits
GLOBAL_MAX_EXAMPLES = 70000

# Random seed for reproducibility
RANDOM_SEED = 42
