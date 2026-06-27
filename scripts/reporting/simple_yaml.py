"""Small stdlib YAML reader for saved Hydra config files.

This is intentionally narrower than a general YAML parser. Reporting only needs
resolved ``.hydra/config.yaml`` files and dotted-path lookups, so supporting the
common Hydra scalar, mapping, and list shapes keeps the cluster-side reporting
CLI free of non-standard runtime dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


ParsedYaml = dict[str, Any]
Line = tuple[int, str]


def load_simple_yaml(path: Path) -> ParsedYaml:
    """Load a practical subset of YAML into Python containers."""
    lines = _logical_lines(Path(path).read_text(encoding="utf-8"))
    if not lines:
        return {}
    data, next_index = _parse_block(lines, 0, lines[0][0])
    if next_index != len(lines):
        _, content = lines[next_index]
        raise ValueError(f"Could not parse YAML line: {content}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in YAML file: {path}")
    return data


def select_path(cfg: ParsedYaml, dotted_path: str, default: Any = None) -> Any:
    """Select a value from nested dictionaries/lists using a dotted path."""
    current: Any = cfg
    for token in dotted_path.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if isinstance(current, list) and token.isdigit():
            index = int(token)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return default
    return current


def _logical_lines(raw_text: str) -> list[Line]:
    lines: list[Line] = []
    for raw_line in raw_text.splitlines():
        without_comment = _strip_comment(raw_line).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        lines.append((indent, without_comment.strip()))
    return lines


def _parse_block(lines: Sequence[Line], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    current_indent, content = lines[index]
    if current_indent < indent:
        return {}, index
    if current_indent > indent:
        raise ValueError(f"Unexpected indentation before YAML line: {content}")
    if content.startswith("-"):
        return _parse_list(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: Sequence[Line], index: int, indent: int) -> tuple[ParsedYaml, int]:
    result: ParsedYaml = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation before YAML line: {content}")
        if content.startswith("-"):
            break

        key, value_text = _split_key_value(content)
        index += 1
        if value_text == "":
            value, index = _parse_nested_or_none(lines, index, current_indent)
        else:
            value = _parse_inline_value(value_text)
        result[key] = value
    return result, index


def _parse_list(lines: Sequence[Line], index: int, indent: int) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation before YAML line: {content}")
        if not content.startswith("-"):
            break

        item_text = content[1:].strip()
        index += 1
        if item_text == "":
            value, index = _parse_nested_or_none(lines, index, current_indent)
            result.append(value)
            continue

        key_value = _try_split_key_value(item_text)
        if key_value is None:
            value = _parse_inline_value(item_text)
            if _has_nested_child(lines, index, current_indent):
                nested, index = _parse_block(lines, index, lines[index][0])
                if isinstance(value, dict) and isinstance(nested, dict):
                    value.update(nested)
                else:
                    value = nested
            result.append(value)
            continue

        key, value_text = key_value
        item: ParsedYaml = {}
        if value_text == "":
            value, index = _parse_nested_or_none(lines, index, current_indent)
            item[key] = value
        else:
            item[key] = _parse_inline_value(value_text)
        if _has_nested_child(lines, index, current_indent):
            nested, index = _parse_block(lines, index, lines[index][0])
            if isinstance(nested, dict):
                item.update(nested)
            else:
                raise ValueError(f"Expected mapping continuation for list item '{key}'.")
        result.append(item)
    return result, index


def _parse_nested_or_none(
    lines: Sequence[Line],
    index: int,
    parent_indent: int,
) -> tuple[Any, int]:
    if _has_nested_child(lines, index, parent_indent):
        return _parse_block(lines, index, lines[index][0])
    return None, index


def _has_nested_child(lines: Sequence[Line], index: int, parent_indent: int) -> bool:
    return index < len(lines) and lines[index][0] > parent_indent


def _split_key_value(content: str) -> tuple[str, str]:
    result = _try_split_key_value(content)
    if result is None:
        raise ValueError(f"Expected YAML key/value line, got: {content}")
    return result


def _try_split_key_value(content: str) -> tuple[str, str] | None:
    colon_index = _find_key_separator(content)
    if colon_index < 0:
        return None
    key = content[:colon_index].strip()
    if not key:
        raise ValueError(f"Expected non-empty YAML key in line: {content}")
    return key, content[colon_index + 1 :].strip()


def _parse_inline_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_inline_value(part) for part in _split_top_level(inner, ",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        parsed: ParsedYaml = {}
        for part in _split_top_level(inner, ","):
            key, raw_item_value = _split_key_value(part)
            parsed[_unquote(key)] = _parse_inline_value(raw_item_value)
        return parsed
    return _parse_scalar(value)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    unquoted = _unquote(value)
    if unquoted != value:
        return unquoted
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index]
    return line


def _split_top_level(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[{(":
            depth += 1
            continue
        if char in "]})":
            depth -= 1
            continue
        if char == separator and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _find_unquoted_separator(value: str, separator: str) -> int:
    quote: str | None = None
    depth = 0
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[{(":
            depth += 1
            continue
        if char in "]})":
            depth -= 1
            continue
        if char == separator and depth == 0:
            return index
    return -1


def _find_key_separator(value: str) -> int:
    quote: str | None = None
    depth = 0
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "[{(":
            depth += 1
            continue
        if char in "]})":
            depth -= 1
            continue
        if char == ":" and depth == 0:
            if index + 1 == len(value) or value[index + 1].isspace():
                return index
    return -1


def _unquote(value: str) -> str:
    if len(value) < 2:
        return value
    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
