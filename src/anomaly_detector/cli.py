"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load (or synthesize) a
price series, engineer causal features, run the two detectors under a strictly
causal train/OOS split, and print the descriptive agreement summary. Typer is
imported lazily and the app is built inside :func:`build_app`, so importing this
module has no side effects (no command registration or I/O at import time). The
module-level :func:`app` is the console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``scan`` and ``demo``) on a fresh
    ``typer.Typer`` instance. Typer is imported lazily inside this function so
    importing :mod:`anomaly_detector.cli` does not import Typer or register
    commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def run(**kwargs: Any) -> int:
    """Run the anomaly scan from the command line.

    Orchestrates: load/synthesize prices -> compute returns -> engineer causal
    features -> causal train/OOS split -> fit each detector on TRAIN, score the
    disjoint OOS slice -> compute descriptive agreement (Jaccard, proxy
    precision/recall, regime alignment) -> emit the honest summary.

    Parameters
    ----------
    **kwargs:
        Parsed command-line options (ticker, date range, detector, contamination,
        window, data-source preference, seed). The concrete signature is bound
        when the Typer command is registered in :func:`build_app`.

    Returns
    -------
    int
        A process exit code (``0`` on success).

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def _demo_defaults() -> dict[str, Any]:
    """Return the option defaults for the offline ``demo`` command.

    A single source of truth for the deterministic, no-network demo run
    (synthetic injected series, ``detector="both"``), shared by the Typer
    command and any test that exercises the demo path.

    Returns
    -------
    dict[str, Any]
        The demo option mapping.
    """
    return {
        "ticker": "SYN",
        "start": date(2015, 1, 1),
        "end": date(2022, 12, 31),
        "detector": "both",
        "contamination": 0.02,
        "window": 21,
        "data_source_pref": "synthetic",
        "seed": 7,
    }


def app() -> None:
    """Console-script entry point for the ``anomaly-detector`` command.

    Builds the Typer app via :func:`build_app` and invokes it. Referenced by
    ``[project.scripts]`` in ``pyproject.toml``.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring (delegates to
        :func:`build_app`).
    """
    build_app()()
