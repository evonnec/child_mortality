"""Every ``child-mortality`` command and Python snippet in README.md runs on the bundled data."""

import re
import shlex
import textwrap
from pathlib import Path

import polars as pl
import pytest

from child_mortality.cli import main

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PROG = ["uv", "run", "child-mortality"]


def code_blocks(language: str) -> list[str]:
    """Fenced blocks in ``language``, including ones indented under a list item."""
    pattern = rf"^( *)```{language}\n(.*?)^\1```"
    found = re.findall(pattern, README, flags=re.MULTILINE | re.DOTALL)
    return [textwrap.dedent(body) for _, body in found]


def cli_commands() -> list[list[str]]:
    """The README's ``uv run child-mortality ...`` commands, continuation lines joined."""
    commands = []
    for block in code_blocks("sh"):
        for line in block.replace("\\\n", " ").splitlines():
            words = shlex.split(line)
            if words[: len(PROG)] == PROG:
                commands.append(words[len(PROG) :])
    return commands


def test_readme_has_cli_examples() -> None:
    assert {command[0] for command in cli_commands()} == {"cd-ceb", "ratio-cube"}


@pytest.mark.parametrize("argv", cli_commands(), ids=lambda argv: " ".join(argv)[:60])
def test_readme_cli_command_runs(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(ROOT)  # the examples use paths relative to the repository root
    argv = list(argv)
    output = argv.index("-o") + 1 if "-o" in argv else None
    if output is not None:
        argv[output] = str(tmp_path / argv[output])  # never write into the checkout
    assert main(argv) == 0
    if output is not None:
        written = Path(argv[output])
        cube = pl.read_parquet(written) if written.suffix == ".parquet" else pl.read_csv(written)
        assert cube.height > 0


@pytest.mark.parametrize("source", code_blocks("python"), ids=lambda s: s.splitlines()[0][:60])
def test_readme_python_snippet_runs(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    namespace: dict[str, object] = {}
    exec(compile(source, "README.md", "exec"), namespace)
    assert isinstance(namespace["cube"], pl.DataFrame)
    assert namespace["cube"].height > 0
