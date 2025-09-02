# Changelog

All notable changes to this dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-20

### Added

- Initial release of the Tool Calling SFT Mix dataset
- 39,538 total examples from multiple sources:
  - xLAM: 20,000 examples
  - OpenFunctions: 11,538 examples
  - Dolly-15k: 4,000 examples
  - WikiText-103: 4,000 examples
- Train/validation/test splits (80%/10%/10%)
- Comprehensive documentation and usage examples
- Sample preview in samples/preview.jsonl
- Validation code and training example
- Dataset card with proper metadata
- Source attribution and licensing information

### Technical Details

- All random operations use seed=42 for reproducibility
- JSON fields are validated and normalized
- Consistent schema across all examples
- Proper handling of tool call formats
- Clean metadata without internal formatting columns
