from typing import Any, Optional


def get_nested_value(data: dict, key_path: str, default: None) -> Any:
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Parameters:
        data: The dictionary to search.
        key_path: A dot-separated string representing the path to the value.
        default: The value to return if the key path does not exist.
    """
    keys = key_path.split('.')
    result = data

    try:
        for key in keys:
            result = result[key]
            if isinstance(result, list) and result:
                result = result[0]
        return result
    except (KeyError, TypeError, IndexError):
        return default


def _validate(data: dict, key_path: str, value: Any) -> bool:
    """
    Validates if a specific key in the dictionary has a specific value.

    Parameters:
        data: The dictionary to validate.
        key_path: A dot-separated string representing the path to the value.
        value: The value to check against.

    Returns:
        bool: True if the value matches, False otherwise.
    """
    return get_nested_value(data, key_path) == value


def is_valid_status(data: dict) -> bool:
    return _validate(data, "status", "Actual")


def is_valid_msg_type(data: dict) -> bool:
    return _validate(data, "msgType", "Alert")


def is_valid_scope(data: dict) -> bool:
    return _validate(data, "scope", "Public")


def is_valid_language(data: dict) -> bool:
    is_valid: Optional[str] = get_nested_value(data, "language")
    return is_valid is None or is_valid.startswith("en")
