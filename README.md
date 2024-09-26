
# Source Code Function Prototype Parser

This script parses source code files to extract function prototypes, generating a summary report of the findings.  It supports multiple programming languages and utilizes the `tree-sitter` library for parsing.

## Features

* **Multi-language Support:** Parses Python, C, C++, Go, Java, JavaScript, Rust, and Bash.
* **Concurrent Processing:** Uses `asyncio` and `concurrent.futures` for efficient parallel processing of files.
* **Caching:** Caches parsed results to speed up subsequent runs.
* **Configurable:** Uses a YAML configuration file for easy customization of input/output paths, logging settings, and supported languages.
* **Robust Error Handling:** Includes detailed error logging and handling of exceptions.
* **Multiple Output Formats:** Generates reports in JSON, CSV, TXT, and XML formats.

## Requirements

* Python 3.7+
* `tree-sitter` (and the necessary language parsers)
* `PyYAML`
* `aiofiles` (for asynchronous file I/O)

## Installation

1. Install required Python packages:
   ```bash
   pip install tree-sitter PyYAML aiofiles
   ```
2. Install and build `tree-sitter` language parsers for the desired languages (refer to the `tree-sitter` documentation).  Place the resulting `languages.dll` (Windows) or `languages.so` (Linux/macOS) file in the same directory as the script.

## Configuration (parser_config.yaml)

The script uses a YAML configuration file (e.g., `parser_config.yaml`) to specify:

* `lib_path`: (Optional) Absolute path to the `tree-sitter` shared library (languages.dll or languages.so).
* `source_directory`: Path to the directory containing source code files.
* `cache_file`: Path to the cache file (pickle format).
* `extensions`: List of file extensions to parse (e.g., ["py", "c", "cpp"]).
* `output`:  Dictionary specifying the output format ("json", "csv", "txt", "xml") and file path.
* `logging`: Dictionary specifying logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), file path, max file size, and backup count.

## Usage

1.  Place your source code files in the directory specified by `source_directory` in `parser_config.yaml`.
2.  Run the script:
    ```bash
    python parse_functions.py parser_config.yaml 
    ```
3. The output report will be written to the file specified in the `output` section of the config file.  Log messages will be written to the file specified in the `logging` section.

## Notes

* Large files (larger than 1MB by default) are skipped to prevent excessive processing time.
* Error messages are logged to the specified log file and also captured in the output report.
