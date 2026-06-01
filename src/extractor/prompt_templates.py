from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


PROMPT_DIR_ENV = "CLIPINATOR_PROMPT_DIR"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts" / "extractor"
_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def prompt_dir() -> Path:
    configured = os.environ.get(PROMPT_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_PROMPT_DIR


def prompt_template_path(name: str) -> Path:
    path = prompt_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {path}. Set {PROMPT_DIR_ENV} to override the prompt directory."
        )
    return path


def load_prompt_template(name: str) -> str:
    return prompt_template_path(name).read_text(encoding="utf-8")


def load_prompt_lines(name: str) -> list[str]:
    return [line.strip() for line in load_prompt_template(name).splitlines() if line.strip()]


def render_prompt_template(name: str, **values: Any) -> str:
    template = load_prompt_template(name)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"Missing prompt template value {key!r} for {name}")
        return str(values[key])

    return _PLACEHOLDER_RE.sub(replace, template)
