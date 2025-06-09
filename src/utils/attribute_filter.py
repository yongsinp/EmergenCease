from typing import Any, Optional

__all__ = ["get_nested_value", "is_valid_alert"]

from src.data.enums import Language


def get_nested_value(data: dict, key_path: str, default: Any = None) -> Any:
    """
    Retrieves a nested value from a dictionary using a dot-separated key path.

    Parameters:
        data: The dictionary to search.
        key_path: A dot-separated string representing the path to the value.
        default: The value to return if the key path does not exist.

    Returns:
        The value at the specified key path.
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
    """Checks if the status of the alert is 'Actual'."""
    return _validate(data, "status", "Actual")


def is_valid_msg_type(data: dict) -> bool:
    """Checks if the message type of the alert is 'Alert'."""
    return _validate(data, "msgType", "Alert")


def is_valid_scope(data: dict) -> bool:
    """Checks if the scope of the alert is 'Public'."""
    return _validate(data, "scope", "Public")


def is_valid_language(data: dict) -> bool:
    """ Checks if the language of the alert is English('en')."""
    is_valid: Optional[str] = get_nested_value(data, "language")
    return is_valid is None or is_valid.startswith(Language.en)


def is_valid_alert(data: dict) -> bool:
    """
    Validates if the alert is valid based on its status, message type, scope, and language.

    Parameters:
        data: The dictionary representing the alert.

    Returns:
        bool: True if the alert is valid, False otherwise.
    """
    return (
            is_valid_status(data)
            and is_valid_msg_type(data)
            and is_valid_scope(data)
            and is_valid_language(data)
    )
