"""Apply structured config patches to run TOML files."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any


def _parse_section_header(line: str) -> str | None:
    m = re.match(r"^\[([^\]]+)\]\s*$", line.strip())
    return m.group(1) if m else None


def _format_toml_value(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return repr(val)
    if isinstance(val, str):
        return repr(val)
    if isinstance(val, list):
        inner = ", ".join(_format_toml_value(v) for v in val)
        return f"[{inner}]"
    return repr(val)


def apply_structured_patch(config_path: Path, patch: dict[str, Any]) -> None:
    """
    Patch config using dotted keys (e.g. graph.edges.top_k) or section dicts.
    Updates existing keys in-place when possible; appends new keys to section.
    """
    if not patch:
        return
    flat: dict[tuple[str, str], Any] = {}
    for k, v in patch.items():
        if "." in k:
            section, key = k.rsplit(".", 1)
            if "." in section:
                # nested section like graph.edges
                pass
            else:
                section = k.rsplit(".", 1)[0]
                key = k.rsplit(".", 1)[1]
            # handle graph.edges.top_k -> section graph.edges, key top_k
            parts = k.split(".")
            if len(parts) >= 2:
                section = ".".join(parts[:-1])
                key = parts[-1]
                flat[(section, key)] = v
        elif isinstance(v, dict):
            for sk, sv in v.items():
                flat[(k, sk)] = sv

    if not flat:
        return

    lines = config_path.read_text(encoding="utf-8").splitlines()
    updated_keys: set[tuple[str, str]] = set()

    current_section: str | None = None
    i = 0
    while i < len(lines):
        hdr = _parse_section_header(lines[i])
        if hdr is not None:
            current_section = hdr
            i += 1
            continue
        if current_section is not None:
            m = re.match(r"^(\s*)([A-Za-z0-9_]+)\s*=", lines[i])
            if m:
                key = m.group(2)
                if (current_section, key) in flat:
                    indent = m.group(1)
                    lines[i] = f"{indent}{key} = {_format_toml_value(flat[(current_section, key)])}"
                    updated_keys.add((current_section, key))
        i += 1

    # append missing keys per section
    by_section: dict[str, list[tuple[str, Any]]] = {}
    for (section, key), val in flat.items():
        if (section, key) not in updated_keys:
            by_section.setdefault(section, []).append((key, val))

    if by_section:
        if lines and lines[-1].strip():
            lines.append("")
        for section in sorted(by_section.keys()):
            header = f"[{section}]"
            if header not in "\n".join(lines):
                lines.append(header)
            for key, val in by_section[section]:
                lines.append(f"{key} = {_format_toml_value(val)}")
            lines.append("")

    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def patch_to_flat_dict(patch: dict[str, Any]) -> dict[str, Any]:
    """Normalize API patch {graph.edges.top_k: 11} for diff preview."""
    out: dict[str, Any] = {}
    for k, v in patch.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"{k}.{sk}"] = sv
        else:
            out[k] = v
    return out


def load_toml_dict(config_path: Path) -> dict[str, Any]:
    return tomllib.loads(config_path.read_text(encoding="utf-8"))
