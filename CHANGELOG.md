# Changelog: Smart-OCR Evolution

## [2026-05-30] - Perfect Accuracy Milestone
### Added
- **Micro-Slicing Strategy**: Implemented row-by-row extraction (6 strips per section) to eliminate spatial hallucinations in the 4B model.
- **Asymmetric Overlap**: Added `s_top (-5px)` and `s_bottom (-15px)` margins to ensure perfect handwriting capture without inter-row bleed.
- **Dynamic Boundary Detection**: Implemented OpenCV Morphological filtering to automatically find the grid start in Slice 1, bypassing printed markers/headers.
- **Worker Failover System**: Added automatic health checks to prioritize the Primary Remote worker and fallback to the Laptop Backup if offline.
- **Metadata-Rich Labels**: Enhanced identification to capture section length (e.g., "2,40 dł") and integrate it into the JSON keys.
- **Global Debug Toggle**: Added `DEBUG_MODE` in `ocr_config.py` to silence terminal output and disable disk I/O for production.
- **RAM-Only Workflow**: Completed the transition to zero-disk storage; all image processing and data transmission happens in RAM buffers.

### Improved
- **Prompt Engineering**: Refined prompts to use 'Inclusive Scan' logic, forcing the model to transcribe every number including natural repetitions.
- **Penalty Tuning**: Optimized `frequency_penalty` and `presence_penalty` (set to 0.0) to allow valid duplicate measurements.
- **Geometric Standardization**: Unified row heights across the document based on total image height (4.4% master ratio).

### Fixed
- Fixed `IndentationError` in `ai.py` logic.
- Resolved "infinite loops" of numbers at the end of grids using the `<EOF>` stop sequence and Tail-Loop Killer.
- Fixed vertical drift issues where Slice 1 was consistently too high and Slice 3 too low.
