#!/usr/bin/env python
"""Serve the static dashboard (wrapper around http.server).

    python scripts/serve_dashboard.py --port 8000
"""

import argparse
import functools
import http.server
import os
import socketserver
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DASHBOARD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dashboard")
)


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the a3ps dashboard.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dir", default=DASHBOARD_DIR, help="Directory to serve.")
    args = p.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.dir)
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"Serving {args.dir} at http://localhost:{args.port}/ (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
