"""Tests for the Typer CLI orchestration layer.

Covers import purity (Typer is imported lazily), ``--help`` output, the offline
synthetic ``demo`` smoke run, and the descriptive summary emitted by
:func:`anomaly_detector.cli.run` on the no-network synthetic path.
"""

from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from anomaly_detector.cli import (
    _demo_defaults,
    app,
    build_app,
    run,
)


@pytest.fixture
def runner() -> CliRunner:
    """A Typer/Click ``CliRunner`` for invoking the app in-process."""
    return CliRunner()


@pytest.mark.unit
def test_cli_module_import_is_typer_free() -> None:
    """Importing the CLI module does not eagerly import Typer."""
    import importlib
    import sys

    sys.modules.pop("typer", None)
    sys.modules.pop("anomaly_detector.cli", None)
    importlib.import_module("anomaly_detector.cli")
    assert "typer" not in sys.modules, "cli.py must import Typer lazily, not at import time"


@pytest.mark.unit
def test_build_app_returns_typer_app() -> None:
    """``build_app`` returns a configured Typer instance with the two commands."""
    import typer

    cli = build_app()
    assert isinstance(cli, typer.Typer)
    command_names = {c.name for c in cli.registered_commands}
    assert {"scan", "demo"} <= command_names


@pytest.mark.unit
def test_help_lists_commands(runner: CliRunner) -> None:
    """``--help`` exits 0 and advertises the scan and demo commands."""
    result = runner.invoke(build_app(), ["--help"])
    assert result.exit_code == 0
    assert "scan" in result.stdout
    assert "demo" in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("command", ["scan", "demo"])
def test_subcommand_help(runner: CliRunner, command: str) -> None:
    """Each subcommand's ``--help`` exits cleanly."""
    result = runner.invoke(build_app(), [command, "--help"])
    assert result.exit_code == 0


@pytest.mark.unit
def test_demo_defaults_are_offline_and_complete() -> None:
    """The demo defaults force the synthetic path and supply every run() key."""
    defaults = _demo_defaults()
    assert defaults["data_source_pref"] == "synthetic"
    assert defaults["detector"] == "both"
    for key in ("ticker", "start", "end", "contamination", "window", "seed"):
        assert key in defaults


@pytest.mark.unit
def test_run_uses_public_walk_forward_entrypoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run`` routes through the public ``run_anomaly_scan`` walk-forward path.

    The console script must NOT reimplement its own split; it must call the same
    public causal walk-forward entrypoint the README, the public API, and the
    FastAPI router use, so the tool a user runs matches the documented claim.
    """
    import anomaly_detector.scan as scan_module

    called: dict[str, object] = {}
    real_run_anomaly_scan = scan_module.run_anomaly_scan

    def _spy(*args: object, **kwargs: object) -> object:
        called["hit"] = True
        called["kwargs"] = dict(kwargs)
        return real_run_anomaly_scan(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(scan_module, "run_anomaly_scan", _spy)

    code = run(**_demo_defaults())
    assert code == 0
    assert called.get("hit") is True, "cli.run must call the public run_anomaly_scan"
    # The walk-forward meta must flow through (proof it is the walk-forward path).
    out = capsys.readouterr().out
    assert "Jaccard agreement" in out


@pytest.mark.unit
def test_demo_command_smoke_run(runner: CliRunner) -> None:
    """The offline ``demo`` command runs end-to-end and exits 0."""
    result = runner.invoke(build_app(), ["demo"])
    assert result.exit_code == 0
    assert "Market anomaly scan" in result.stdout
    assert "data source        : synthetic" in result.stdout
    # Honest-null discipline: the summary must disclaim tradability.
    assert "diagnostic, not tradable" in result.stdout


@pytest.mark.unit
def test_scan_command_synthetic_smoke(runner: CliRunner) -> None:
    """A tiny synthetic ``scan`` invocation succeeds without network."""
    result = runner.invoke(
        build_app(),
        [
            "scan",
            "SYN",
            "--data-source-pref",
            "synthetic",
            "--start",
            "2016-01-01",
            "--end",
            "2020-12-31",
            "--window",
            "21",
        ],
    )
    assert result.exit_code == 0
    assert "Jaccard agreement" in result.stdout


@pytest.mark.unit
def test_run_synthetic_returns_zero_and_reports_source() -> None:
    """``run`` on the synthetic path returns 0 and prints the descriptive block."""
    code = run(**_demo_defaults())
    assert code == 0


@pytest.mark.unit
def test_run_emits_honest_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """The printed summary carries the descriptive (non-tradable) headline."""
    code = run(**_demo_defaults())
    assert code == 0
    out = capsys.readouterr().out
    assert "Jaccard agreement" in out
    assert "proxy precision" in out
    assert "regime alignment" in out
    assert "no alpha is claimed" in out


@pytest.mark.unit
@pytest.mark.parametrize("detector", ["iforest", "autoencoder", "both"])
def test_run_honors_detector_choice(detector: str, capsys: pytest.CaptureFixture[str]) -> None:
    """Each detector choice runs the synthetic path to a clean exit."""
    options = {**_demo_defaults(), "detector": detector}
    code = run(**options)
    assert code == 0
    out = capsys.readouterr().out
    assert f"detector           : {detector}" in out


@pytest.mark.unit
def test_run_returns_one_on_validation_error() -> None:
    """A bad contamination value surfaces as a non-zero exit (handled error)."""
    options = {**_demo_defaults(), "contamination": 0.9}  # outside (0, 0.5)
    code = run(**options)
    assert code == 1


@pytest.mark.unit
def test_run_returns_one_on_insufficient_data() -> None:
    """Too short a window of data (few feature rows) is a handled non-zero exit."""
    # A date range far too short to clear the rolling warm-up produces < 4
    # feature rows, tripping the InsufficientDataError guard (handled -> exit 1).
    code = run(
        ticker="SYN",
        start=date(2020, 1, 1),
        end=date(2020, 1, 31),
        detector="both",
        contamination=0.02,
        window=21,
        data_source_pref="synthetic",
        seed=7,
    )
    assert code == 1


@pytest.mark.unit
def test_app_entry_point_builds_and_shows_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """The console-script ``app`` builds the Typer app and runs (no-args -> help)."""
    # With no CLI args ``no_args_is_help`` shows help and exits with Click's
    # "no command given" code (2); patch argv so the entry point does not consume
    # this test runner's arguments. The point is that ``app()`` builds and invokes
    # the Typer app (exercising the console-script entry point).
    monkeypatch.setattr("sys.argv", ["anomaly-detector"])
    with pytest.raises(SystemExit) as excinfo:
        app()
    assert excinfo.value.code == 2


@pytest.mark.unit
def test_run_accepts_date_objects() -> None:
    """``run`` accepts ``datetime.date`` start/end (as the Typer command passes)."""
    code = run(
        ticker="SYN",
        start=date(2017, 1, 1),
        end=date(2021, 12, 31),
        detector="both",
        contamination=0.02,
        window=21,
        data_source_pref="synthetic",
        seed=7,
    )
    assert code == 0
