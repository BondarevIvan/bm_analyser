from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SUMMATOR = (ROOT / "examples" / "summator.mgn").read_text(encoding="utf-8")


def wrap(body: str, name: str = "test") -> str:
    """Обернуть фрагмент (события/функции/controls) в model."""
    return f'model "{name}" {{\n{body}\n}}'


@pytest.fixture
def summator_text():
    return SUMMATOR
