import os
import platform
import importlib.util
import ctypes
import asyncio
import pickle
import hashlib
import logging
import argparse
import json
import csv
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language, get_parser
from time import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import yaml
from logging.handlers import MemoryHandler, RotatingFileHandler
import traceback

# Constants
UNKNOWN_FUNCTION_NAME = "unknown"
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE_MAX_BYTES = 1048576  # 1 MB for rotating log files
DEFAULT_LOG_FILE_BACKUP_COUNT = 3
BATCH_SIZE = 10  # Batch size for concurrent file processing
MAX_FILE_SIZE = 1048576  # 1 MB max size for files to process

# Supported file extensions and their mappings to Tree-sitter language names
EXTENSIONS = {
    'python': 'py',
    'c': 'c',
    'cpp': 'cpp',
    'go': 'go',
    'java': 'java',
    'javascript': 'js',
    'rust': 'rs',
    'bash': 'sh'
}

# Language configuration for Tree-sitter parsing
SUPPORTED_LANGUAGES = {
    "py": {
        "identifier_node": "identifier",
        "function_definition_node": "function_definition",
        "parameters_node": "parameters",
        "return_type_node": None,
        "body_node": "block"
    },
    "c": {
        "identifier_node": "identifier",
        "function_definition_node": "function_definition",
        "parameters_node": "parameter_list",
        "return_type_node": "type",
        "body_node": "compound_statement"
    },
    "cpp": {
        "identifier_node": "identifier",
        "function_definition_node": "function_definition",
        "parameters_node": "parameter_list",
        "return_type_node": "type",
        "body_node": "compound_statement"
    },
    "go": {
        "identifier_node": "identifier",
        "function_definition_node": "function_declaration",
        "parameters_node": "parameter_list",
        "return_type_node": "result",
        "body_node": "block"
    },
    "java": {
        "identifier_node": "identifier",
        "function_definition_node": "method_declaration",
        "parameters_node": "formal_parameters",
        "return_type_node": "type",
        "body_node": "block"
    },
    "js": {
        "identifier_node": "identifier",
        "function_definition_node": "function_declaration",
        "parameters_node": "formal_parameters",
        "return_type_node": None,
        "body_node": "statement_block"
    },
    "rs": {
        "identifier_node": "identifier",
        "function_definition_node": "function_item",
        "parameters_node": "parameters",
        "return_type_node": "return_type",
        "body_node": "block"
    },
    "sh": {
        "identifier_node": "identifier",
        "function_definition_node": "function_definition",
        "parameters_node": "parameters",
        "return_type_node": None,
        "body_node": "compound_statement"
    }
}

class FunctionPrototype:
    """Represents a function prototype with its signature and unique hash."""
    def __init__(self, name: str, arg_types: List[str], return_types: List[str], body_symbols: List[str] = None):
        self.name = name
        self.arg_types = arg_types
        self.return_types = return_types
        self.body_symbols = body_symbols or []

    def to_signature(self) -> str:
        """Returns the string representation of the function signature."""
        return f"{self.name}({', '.join(self.arg_types)}) -> {', '.join(self.return_types)}"

    def unique_hash(self) -> str:
        """Generates a unique hash for the function prototype."""
        combined_string = self.to_signature() + "".join(self.body_symbols)
        sha256_hash = hashlib.sha256(combined_string.encode()).hexdigest()
        return sha256_hash

def get_parser(language_name: str, lib_path: str = None) -> Optional[Parser]:
    """Creates a Tree-sitter parser for the specified language."""
    if lib_path is None:
        if platform.system().lower() == 'windows':
            lib_path = str(pathlib.Path(__file__).parent / "my_languages.dll")
        else:
            lib_path = str(pathlib.Path(__file__).parent / "my_languages.so")

    try:
        language = Language(lib_path, language_name)
        parser = Parser()
        parser.set_language(language)
        return parser
    except (FileNotFoundError, Exception) as e:
        logging.error(f"Error creating parser for '{language_name}': {e}. Traceback: {traceback.format_exc()}")
        return None
        
async def read_file(file_path: str, executor: ThreadPoolExecutor) -> str:
    """Asynchronously reads the content of a file."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(executor, lambda: open(file_path, 'r', encoding='utf-8').read())
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}. Traceback: {traceback.format_exc()}")
        raise

def _extract_symbols_from_body(body_node: Optional[Any], language_config: Dict[str, Any]) -> List[str]:
    """Extracts relevant symbols (identifiers) from the function body using Tree-sitter."""
    identifier_type = language_config.get("identifier_node")
    if not identifier_type or not body_node:
        return []

    symbols = []
    try:
        for node in body_node.walk():
            if node.type == identifier_type:
                symbols.append(node.text.decode("utf-8"))
    except Exception as e:
        logging.error(f"Error extracting symbols from body. Traceback: {traceback.format_exc()}")
    return symbols

def parse_prototypes(source_code: str, parser, language_config: Dict[str, Any]) -> List[FunctionPrototype]:
    """Parses function prototypes from the given source code using Tree-sitter."""
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node
        prototypes = []

        for child in root_node.children:
            if child.type == language_config["function_definition_node"]:
                func_name_node = child.child_by_field_name("name")
                params_node = child.child_by_field_name(language_config["parameters_node"])
                return_type_node = child.child_by_field_name(language_config["return_type_node"])
                body_node = child.child_by_field_name(language_config["body_node"])

                name = func_name_node.text.decode("utf-8") if func_name_node else UNKNOWN_FUNCTION_NAME
                arg_types = [param.text.decode("utf-8") for param in params_node.children] if params_node else []
                return_types = [return_type_node.text.decode("utf-8")] if return_type_node else []
                body_symbols = _extract_symbols_from_body(body_node, language_config)

                prototypes.append(FunctionPrototype(name, arg_types, return_types, body_symbols))

        return prototypes
    except Exception as e:
        logging.error(f"Error parsing source code: {e}. Traceback: {traceback.format_exc()}")
        return []

def load_cache(cache_file: str) -> Dict[str, Any]:
    """Loads function prototypes from the cache file."""
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading cache file {cache_file}: {e}. Traceback: {traceback.format_exc()}")
        return {}

def save_cache(cache_file: str, data: Dict[str, Any]):
    """Saves function prototypes to the cache file."""
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logging.error(f"Error saving cache to file {cache_file}: {e}. Traceback: {traceback.format_exc()}")

async def parse_source_file(file_path: str, parser, cache: Dict[str, Any], language_config: Dict[str, Any], executor: ThreadPoolExecutor) -> List[FunctionPrototype]:
    """Parses function prototypes from a single source file."""
    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE:
            logging.warning(f"File {file_path} is too large to parse. Skipping...")
            return []

        file_hash_value = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
        if file_hash_value in cache:
            return cache[file_hash_value]

        source_code = await read_file(file_path, executor)
        with ProcessPoolExecutor() as process_executor:
            prototypes = await asyncio.get_event_loop().run_in_executor(
                process_executor, parse_prototypes, source_code, parser, language_config
            )

        if prototypes:
            cache[file_hash_value] = prototypes
        return prototypes
    except Exception as e:
        logging.error(f"Error parsing file {file_path}: {e}. Traceback: {traceback.format_exc()}")
        return []

async def parse_source_files(directory: str, cache_file: str, extensions: List[str], lib_path: str = None) -> tuple[List[FunctionPrototype], int]:
    """Parses function prototypes from all source files in the given directory."""
    cache = load_cache(cache_file)
    unique_prototypes = {}

    tasks = []
    file_count = 0

    with ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(directory):
            for file in files:
                file_extension = os.path.splitext(file)[1][1:].lower()
                if file_extension in extensions:
                    file_path = os.path.join(root, file)
                    language_name = EXTENSIONS.get(file_extension)

                    if language_name:
                        parser = get_parser(language_name, lib_path)   
                        if parser is None:
                            continue
                            
                        language_config = SUPPORTED_LANGUAGES.get(language_name)
                        if not language_config:
                            logging.warning(f"Unsupported file extension: {file_extension}. Skipping {file_path}.")
                            continue

                        tasks.append(parse_source_file(file_path, parser, cache, language_config, executor))
                        file_count += 1

            for i in range(0, len(tasks), BATCH_SIZE):
                batch_tasks = tasks[i:i + BATCH_SIZE]
                try:
                    results = await asyncio.gather(*batch_tasks)
                except Exception as e:
                    logging.error(f"Error during batch processing: {e}. Traceback: {traceback.format_exc()}")
                    continue

                for file_prototypes in results:
                    for prototype in file_prototypes:
                        signature_hash = prototype.unique_hash()
                        unique_prototypes[signature_hash] = prototype

    save_cache(cache_file, unique_prototypes)
    return list(unique_prototypes.values()), file_count

def signature_maker(prototypes: List[FunctionPrototype]) -> set:
    """Creates a set of unique function signatures."""
    return {prototype.to_signature() for prototype in prototypes}

def setup_logging(log_config: Dict[str, Any]):
    """Configures logging settings."""
    log_level = getattr(logging, log_config.get("level", "WARNING").upper(), logging.WARNING)
    log_file = log_config.get("file")

    if log_file:
        handler = RotatingFileHandler(
            log_file,
            maxBytes=log_config.get("max_bytes", DEFAULT_LOG_FILE_MAX_BYTES),
            backupCount=log_config.get("backup_count", DEFAULT_LOG_FILE_BACKUP_COUNT)
        )
        formatter = logging.Formatter(LOGGING_FORMAT)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    logging.basicConfig(level=log_level, format=LOGGING_FORMAT)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Parse source files for function prototypes.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (YAML or JSON)")
    return parser.parse_args()

def generate_summary_report(file_count: int, function_count: int, duration: float, errors: List[str]) -> Dict[str, Any]:
    """Generates a summary report of the parsing process."""
    return {
        "files_processed": file_count,
        "functions_found": function_count,
        "processing_time_seconds": duration,
        "errors_encountered": errors
    }

def write_output(output_format: str, output_file: str, report: Dict[str, Any]):
    """Writes the report in the specified format."""
    output_writers = {
        "json": write_json_output,
        "csv": write_csv_output,
        "txt": write_txt_output,
        "xml": write_xml_output,
    }
    if output_format in output_writers:
        output_writers[output_format](output_file, report)
    else:
        logging.error(f"Unsupported output format: {output_format}")

def write_json_output(output_file: str, report: Dict[str, Any]):
    """Writes report in JSON format."""
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logging.error(f"Error writing JSON output to {output_file}: {e}. Traceback: {traceback.format_exc()}")

def write_csv_output(output_file: str, report: Dict[str, Any]):
    """Writes report in CSV format."""
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=report.keys())
            writer.writeheader()
            writer.writerow(report)
    except Exception as e:
        logging.error(f"Error writing CSV output to {output_file}: {e}. Traceback: {traceback.format_exc()}")

def write_txt_output(output_file: str, report: Dict[str, Any]):
    """Writes report in plain text format."""
    try:
        with open(output_file, 'w') as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        logging.error(f"Error writing TXT output to {output_file}: {e}. Traceback: {traceback.format_exc()}")

def write_xml_output(output_file: str, report: Dict[str, Any]):
    """Writes report in XML format."""
    try:
        root = ET.Element("Report")
        for key, value in report.items():
            child = ET.SubElement(root, key)
            child.text = str(value)
        tree = ET.ElementTree(root)
        tree.write(output_file)
    except Exception as e:
        logging.error(f"Error writing XML output to {output_file}: {e}. Traceback: {traceback.format_exc()}")

def load_config(config_file: str) -> Dict[str, Any]:
    """Loads configuration from a JSON or YAML file."""
    try:
        if config_file.endswith((".yaml", ".yml")):
            return load_yaml_config(config_file)
        elif config_file.endswith(".json"):
            return load_json_config(config_file)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_file}")
    except Exception as e:
        logging.error(f"Error loading configuration file {config_file}: {e}. Traceback: {traceback.format_exc()}")
        raise

def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading YAML config file {config_file}: {e}. Traceback: {traceback.format_exc()}")
        raise

def load_json_config(config_file: str) -> Dict[str, Any]:
    """Loads configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON config file {config_file}: {e}. Traceback: {traceback.format_exc()}")
        raise

async def main():
    """Main asynchronous function to execute the parsing process."""
    args = parse_arguments()
    config = load_config(args.config_file)
    lib_path = config.get("lib_path")  # Get lib_path from config

    # Set up in-memory logging to capture errors
    memory_handler = MemoryHandler(capacity=1000, flushLevel=logging.ERROR)
    logging.getLogger().addHandler(memory_handler)

    # Set up file logging (or other types) based on the config
    setup_logging(config.get("logging", {}))

    start_time = time()

    function_prototypes, file_count = await parse_source_files(
        config["source_directory"],
        config["cache_file"],
        config["extensions"],
        lib_path
    )

    signatures = signature_maker(function_prototypes)

    duration = time() - start_time

    # Capture errors from the in-memory handler
    errors = [record.getMessage() for record in memory_handler.buffer if record.levelno >= logging.ERROR]

    # Generate the summary report
    report = generate_summary_report(file_count, len(signatures), duration, errors)

    # Write the output report
    write_output(config["output"]["format"], config["output"]["file"], report)

    # Flush any remaining logs in the memory handler
    memory_handler.flush()

if __name__ == "__main__":
    asyncio.run(main())