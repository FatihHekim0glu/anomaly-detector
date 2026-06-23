"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load (or synthesize) a
price series and hand it to the public :func:`anomaly_detector.scan.run_anomaly_scan`
walk-forward entrypoint - the SAME causal walk-forward path the README headline,
the public API, and the deployed FastAPI router all use - then print its
descriptive agreement summary. Routing the console script through the public
entrypoint keeps the tool a user runs and the documented walk-forward claim in
lockstep (no divergent inline split). Typer is imported lazily and the app is
built inside :func:`build_app`, so importing this module has no side effects (no
command registration or I/O at import time). The module-level :func:`app` is the
console-script entry point.

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
    """
    # LAZY import: keep Typer off this pure module's import path.
    import typer

    cli = typer.Typer(
        name="anomaly-detector",
        add_completion=False,
        help="Flag anomalous trading days in a liquid ETF with two independent "
        "unsupervised detectors (Isolation Forest + a PCA reconstruction-error "
        "autoencoder) under a strictly causal walk-forward refit. The flags are "
        "diagnostic, not tradable.",
        no_args_is_help=True,
    )

    @cli.command("scan")
    def _scan_command(
        ticker: str = typer.Argument("SPY", help="Asset symbol to scan (e.g. SPY)."),
        start: str = typer.Option("2015-01-01", help="Inclusive start date (YYYY-MM-DD)."),
        end: str = typer.Option("2022-12-31", help="Inclusive end date (YYYY-MM-DD)."),
        detector: str = typer.Option(
            "both", help="Detector to highlight (iforest|autoencoder|both)."
        ),
        contamination: float = typer.Option(
            0.02, help="Expected anomalous fraction (0 < c < 0.5)."
        ),
        window: int = typer.Option(21, help="Rolling feature window (trading days)."),
        data_source_pref: str = typer.Option(
            "auto", help="Data-source preference (auto|polygon|synthetic)."
        ),
        seed: int = typer.Option(7, help="Master RNG seed (deterministic)."),
    ) -> None:
        """Scan a ticker for anomalous days under a strictly causal split."""
        code = run(
            ticker=ticker,
            start=date.fromisoformat(start),
            end=date.fromisoformat(end),
            detector=detector,
            contamination=contamination,
            window=window,
            data_source_pref=data_source_pref,
            seed=seed,
        )
        raise typer.Exit(code=code)

    @cli.command("demo")
    def _demo_command() -> None:
        """Run the scan on a deterministic synthetic injected series (no network)."""
        code = run(**_demo_defaults())
        raise typer.Exit(code=code)

    return cli


def run(**kwargs: Any) -> int:
    """Run the anomaly scan from the command line.

    Orchestrates: load/synthesize prices -> hand the price series to the public
    :func:`anomaly_detector.scan.run_anomaly_scan` walk-forward entrypoint (which
    engineers causal features and runs the anchored/expanding walk-forward refit
    of both detectors, concatenating the disjoint OOS folds into a single
    zero-look-ahead score/flag series) -> emit the honest descriptive summary.

    Using the public entrypoint keeps the CLI on the EXACT causal walk-forward
    path the README headline, the public API, and the deployed FastAPI router all
    use, so the tool a user runs never disagrees with the documented walk-forward
    claim.

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
    """
    # Imports are local so importing this module stays side-effect free and the
    # heavy compute modules are only paid for at invocation time.
    from anomaly_detector._exceptions import AnomalyDetectorError
    from anomaly_detector.data import load_prices
    from anomaly_detector.scan import run_anomaly_scan

    ticker: str = str(kwargs.get("ticker", "SPY"))
    start: date = kwargs["start"]
    end: date = kwargs["end"]
    detector: str = str(kwargs.get("detector", "both"))
    contamination: float = float(kwargs.get("contamination", 0.02))
    window: int = int(kwargs.get("window", 21))
    data_source_pref: str = str(kwargs.get("data_source_pref", "auto"))
    seed: int = int(kwargs.get("seed", 7))

    try:
        # --- Load data (Polygon -> synthetic fallback) --------------------
        prices, data_source = load_prices(
            ticker,
            start,
            end,
            source_pref=data_source_pref,  # type: ignore[arg-type]
            seed=seed,
        )

        # --- Run the PUBLIC causal walk-forward scan ----------------------
        # Both detectors are always fitted inside run_anomaly_scan regardless of
        # the highlighted choice, so the descriptive Jaccard agreement (the
        # headline) is always available.
        scan = run_anomaly_scan(
            prices=prices,
            detector=detector,  # type: ignore[arg-type]
            contamination=contamination,
            window=window,
            seed=seed,
            data_source=data_source,
        )

        result_if = scan.result_iforest
        result_ae = scan.result_autoencoder
        agreement = scan.agreement
        summary = scan.summary()

        # --- Emit the honest, descriptive summary -------------------------
        print("Market anomaly scan")
        print("=" * 40)
        print(f"ticker             : {ticker}")
        print(f"data source        : {data_source}")
        print(f"detector           : {detector}")
        print(f"OOS observations   : {result_if.n_test}")
        print(f"primary flags      : {int(summary['n_flags'])}")
        print(f"iforest flags      : {int(result_if.flags.sum())}")
        print(f"autoencoder flags  : {int(result_ae.flags.sum())}")
        print(f"Jaccard agreement  : {agreement.jaccard:.4f}")
        print(f"proxy precision    : {agreement.proxy_precision:.4f}")
        print(f"proxy recall       : {agreement.proxy_recall:.4f}")
        print(f"regime alignment   : {agreement.regime_alignment:.4f}")
        top_dates = ", ".join(agreement.top_anomaly_dates[:5]) or "(none)"
        print(f"top anomaly dates  : {top_dates}")
        print("-" * 40)
        print(
            "Flags are diagnostic, not tradable - there is no ground-truth "
            "anomaly label, so no alpha is claimed."
        )
    except AnomalyDetectorError as exc:
        print(f"error: {exc}")
        return 1

    return 0


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
    """
    build_app()()
