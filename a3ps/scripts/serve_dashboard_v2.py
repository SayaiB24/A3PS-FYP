#!/usr/bin/env python
"""Serve the v2 results dashboard (read-only). The legacy dashboard is untouched.

    python scripts/serve_dashboard_v2.py --port 8010

Static files come from dashboard_v2/. JSON comes from scripts/dashboard_v2_data.py,
which only READS existing pipeline artifacts (eval/, notebooks/models/,
dashboard/clips/, eval/anticipation/). Every non-GET method is refused.

Endpoints
    GET /api/manifest                the legacy clip manifest (same file, same shapes)
    GET /api/runs                    all discovered runs + their evaluation records
    GET /api/run/<id>                one run incl. per-epoch training history
    GET /api/rows/<eval_id>          per-clip rows for an evaluation that has them
    GET /api/clip/<clip_id>?overlay=full|legacy&hz=10
                                     downsampled replay payload for one clip
    GET /media/video/<clip_id>       the clip's video, with HTTP Range support
"""

import argparse
import functools
import http.server
import json
import os
import re
import shutil
import socketserver
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dashboard_v2_data as data  # noqa: E402

STATIC_DIR = os.path.join(data.ROOT, "dashboard_v2")


class Handler(http.server.SimpleHTTPRequestHandler):
    store: "data.DashboardStore" = None

    def log_message(self, fmt, *args):  # quieter: skip static 200s
        if "/api/" in self.path or (args and str(args[1]) not in ("200", "206", "304")):
            super().log_message(fmt, *args)

    # ---- read-only guard ----
    def _refuse(self):
        self.send_error(405, "read-only dashboard")

    do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

    def _json(self, obj, status=200):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        q = urllib.parse.parse_qs(url.query)
        try:
            if path == "/api/manifest":
                return self._json(self.store.manifest())
            if path == "/api/runs":
                return self._json(self.store.runs_summary())
            m = re.fullmatch(r"/api/run/(.+)", path)
            if m:
                r = self.store.run_detail(m.group(1))
                return self._json(r) if r else self._json({"error": "unknown run"}, 404)
            m = re.fullmatch(r"/api/rows/(.+)", path)
            if m:
                rows = self.store.eval_rows(m.group(1))
                return self._json({"rows": rows}) if rows is not None \
                    else self._json({"error": "no per-clip rows for this evaluation"}, 404)
            m = re.fullmatch(r"/api/clip/([\w\-]+)", path)
            if m:
                hz = float(q.get("hz", ["10"])[0])
                t0 = time.time()
                d = self.store.clip_detail(m.group(1), overlay=q.get("overlay", [None])[0],
                                           hz=max(1.0, min(30.0, hz)))
                d["load_ms"] = round((time.time() - t0) * 1000)
                return self._json(d)
            m = re.fullmatch(r"/media/video/([\w\-]+)", path)
            if m:
                return self._send_video(self.store.video_path(m.group(1)))
        except Exception as e:  # surface, don't hang the UI
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return super().do_GET()

    def do_HEAD(self):
        if self.path.startswith("/media/video/"):
            cid = self.path.rsplit("/", 1)[-1]
            p = self.store.video_path(cid)
            if not p:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(os.path.getsize(p)))
            self.send_header("Accept-Ranges", "bytes")
            return self.end_headers()
        return super().do_HEAD()

    def _send_video(self, p):
        """Stream an mp4 with Range support so the <video> element can seek."""
        if not p or not os.path.isfile(p):
            return self.send_error(404, "no video for this clip")
        size = os.path.getsize(p)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = size - int(m.group(2))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                return self.end_headers()
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(p, "rb") as fh:
            fh.seek(start)
            remaining = length
            try:
                while remaining > 0:
                    chunk = fh.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser aborts range reads routinely while seeking


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    p = argparse.ArgumentParser(description="Serve the a3ps v2 results dashboard (read-only).")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-checkpoints", action="store_true",
                   help="Skip reading checkpoint metadata (no torch import).")
    args = p.parse_args()

    # The reused readers resolve paths relative to the a3ps project root.
    os.chdir(data.ROOT)
    t0 = time.time()
    Handler.store = data.DashboardStore(load_checkpoints=not args.no_checkpoints)
    n_rows = sum(len(v) for v in Handler.store.rows.values())
    print(f"Discovered {len(Handler.store.runs)} runs, {n_rows} per-clip rows "
          f"in {time.time() - t0:.1f}s")
    for w in Handler.store.warnings:
        print("  !", w)
    handler = functools.partial(Handler, directory=STATIC_DIR)
    with ThreadingServer((args.host, args.port), handler) as httpd:
        print(f"Serving {STATIC_DIR} at http://{args.host}:{args.port}/ (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
