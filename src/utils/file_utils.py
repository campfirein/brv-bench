"""File Utilities."""


# =============================================================================

import json

import yaml  # type: ignore

# =============================================================================


def load_yaml(yaml_path: str) -> dict:
    """Load data from YAML file.

    Args:
    ----
    yaml_path (str) : path to input YAML file

    Returns:
    -------
    (dict) : data loaded from YAML file

    """
    data = None
    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    return data


# =============================================================================


def load_json(json_path: str, encoding: str = "utf-8") -> dict:
    """Load data from JSON file.

    Args:
    ----
    json_path (str) : path to input JSON file
    encoding (Str): type of encoding
        (default: `utf-8`)

    Returns:
    -------
    (dict) : data loaded from JSON file

    """
    data = None
    with open(json_path, encoding=encoding) as f:
        data = json.load(f)

    return data


# =============================================================================


def save_json(
    json_path: str, data: dict, indent: int = 2, ensure_ascii: bool = False
) -> None:
    """Serialize data to JSON file.

    Args:
    ----
    json_path (Str): path of output JSON file
    data (Dict): content to serialize
    indent (Int): set default indentation
    ensure_ascii (Bool): set output content is as-is

    """
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


# =============================================================================
