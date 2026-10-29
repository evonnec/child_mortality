"""Command line interface: ``child-mortality cd-ceb`` and ``child-mortality ratio-cube``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

lazy import polars as pl

from child_mortality import __version__
from child_mortality.cube import ratio_cube
from child_mortality.survey import (
    DIMENSIONS,
    NA_VALUES,
    cd_ceb_cube,
)

PROG = "child-mortality"
EXIT_OK = 0
EXIT_ERROR = 2
_WRITERS = (".csv", ".parquet")


class _Parser(argparse.ArgumentParser):
    """Argument parser whose errors are one line, ``child-mortality: error: ...``."""

    def error(self, message: str) -> NoReturn:
        raise SystemExit(fail(message))


def fail(message: str) -> int:
    """Print a one-line error and return the error exit code."""
    print(f"{PROG}: error: {message.strip().splitlines()[0]}", file=sys.stderr)
    return EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog=PROG, description="Ratio metrics for every dimensional cut of a table.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    cd_ceb = commands.add_parser(
        "cd-ceb", help="children died / children ever born from a women and a household file"
    )
    cd_ceb.add_argument("--women", required=True, type=Path, help="women's CSV")
    cd_ceb.add_argument("--households", required=True, type=Path, help="household CSV")
    cd_ceb.add_argument(
        "-d",
        "--dimension",
        action="append",
        dest="dimensions",
        metavar="DIM",
        help="dimension column; repeatable (default: the nine survey dimensions)",
    )
    _add_cube_options(cd_ceb)

    generic = commands.add_parser("ratio-cube", help="sum(numerator) / sum(denominator) of a CSV")
    generic.add_argument("input", type=Path, help="input CSV")
    generic.add_argument(
        "-d",
        "--dimension",
        action="append",
        dest="dimensions",
        metavar="DIM",
        required=True,
        help="dimension column; repeatable",
    )
    generic.add_argument("--numerator", required=True, help="numerator column")
    generic.add_argument("--denominator", required=True, help="denominator column")
    _add_cube_options(generic)
    return parser


def _add_cube_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-dims",
        type=int,
        help="largest cut size, over all listed dimensions (default: all of them)",
    )
    parser.add_argument(
        "--min-rows", type=int, default=1, help="drop cells with fewer rows (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="write .csv or .parquet (default: CSV on stdout)"
    )


def read_generic(path: Path, numerator: str, denominator: str) -> pl.LazyFrame:
    """Read a CSV as String columns, except the two measures (Float64); NA and empty are null."""
    frame = pl.scan_csv(path, infer_schema=False, null_values=list(NA_VALUES))
    return frame.with_columns(pl.col(numerator, denominator).cast(pl.Float64))


def write_cube(cube: pl.DataFrame, output: Path | None) -> None:
    """Write the cube to ``output`` (by suffix), or as CSV to stdout."""
    if output is None:
        sys.stdout.write(cube.write_csv())
    elif output.suffix.lower() == ".parquet":
        cube.write_parquet(output)
    else:
        cube.write_csv(output)


def run(args: argparse.Namespace) -> pl.DataFrame:
    if args.command == "cd-ceb":
        return cd_ceb_cube(
            args.women,
            args.households,
            dimensions=args.dimensions or DIMENSIONS,
            max_dims=args.max_dims,
            min_rows=args.min_rows,
        )
    data = read_generic(args.input, args.numerator, args.denominator)
    return ratio_cube(
        data,
        args.dimensions,
        args.numerator,
        args.denominator,
        max_dims=args.max_dims,
        min_rows=args.min_rows,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code (0 ok, 2 bad arguments, files or data)."""
    args = build_parser().parse_args(argv)
    output: Path | None = args.output
    if output is not None and output.suffix.lower() not in _WRITERS:
        return fail(f"output must end in .csv or .parquet, got {output.name!r}")
    try:
        write_cube(run(args), output)
    except ValueError, OSError, pl.exceptions.PolarsError:
        return fail(str(sys.exception()))
    return EXIT_OK
