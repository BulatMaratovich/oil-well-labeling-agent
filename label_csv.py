#!/usr/bin/env python3
"""
Interactive time-series labeling tool.

Starts a local web server and opens the labeling UI in the browser.

Usage:
    python label_csv.py [CSV_FILE] [SCENARIO_YAML]

Examples:
    python label_csv.py                           # open UI, upload from browser
    python label_csv.py data/my_data.csv          # pre-load CSV on startup
    python label_csv.py data/my_data.csv scenarios/amplitude_anomaly.yaml

The UI will open at http://localhost:8000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path


def check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("       export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)


def check_dependencies() -> None:
    missing = []
    for pkg in ["fastapi", "uvicorn", "anthropic", "numpy", "yaml", "dateutil"]:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def open_browser_delayed(url: str, delay: float = 2.0) -> None:
    """Open browser after a short delay to let the server start."""
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive time-series labeling tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        help="CSV file to label (optional — can be uploaded from UI)",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="scenarios/belt_break.yaml",
        help="Scenario YAML file (default: scenarios/belt_break.yaml)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to run on (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open browser",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    check_api_key()
    check_dependencies()

    # Validate paths if provided
    if args.csv_file and not Path(args.csv_file).exists():
        print(f"ERROR: CSV file not found: {args.csv_file}")
        sys.exit(1)

    if args.scenario and not Path(args.scenario).exists():
        print(f"ERROR: Scenario file not found: {args.scenario}")
        sys.exit(1)

    # If CSV provided, we'll pass it as env vars for the server to pre-load
    if args.csv_file:
        os.environ["PRELOAD_CSV"] = str(Path(args.csv_file).resolve())
    if args.scenario:
        os.environ["PRELOAD_SCENARIO"] = str(Path(args.scenario).resolve())

    url = f"http://localhost:{args.port}"

    print()
    print("=" * 60)
    print("  TIME-SERIES LABELING AGENT")
    print("=" * 60)
    print(f"  URL:      {url}")
    if args.csv_file:
        print(f"  CSV:      {args.csv_file}")
    print(f"  Scenario: {args.scenario}")
    print(f"  API key:  {'✓ set' if os.environ.get('ANTHROPIC_API_KEY') else '✗ missing'}")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    if not args.no_browser:
        open_browser_delayed(url, delay=1.5)

    # Start uvicorn
    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
