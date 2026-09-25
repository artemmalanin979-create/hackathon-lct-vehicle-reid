"""Loopback gateway for capturing current UI against an existing real API.

Only static files and the listed read-only API actions are served. The gateway
never starts/stops a container, alters its collection or writes uploaded data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
import json


GET_ROUTES = {"/api/ui/state", "/api/demo", "/api/materials",
              "/api/health", "/api/version", "/docs", "/openapi.json"}
POST_ROUTES = {"/api/search", "/api/explain"}
FORWARDED_HEADERS = {"Content-Type", "Cache-Control", "Content-Disposition",
                     "X-Content-Type-Options"}
COUNTS: Counter[str] = Counter()


class Gateway(SimpleHTTPRequestHandler):
    upstream: str
    static_root: Path

    def log_message(self, format: str, *args: object) -> None:
        # Browser URLs can contain dataset identifiers. Keep only route counts.
        pass

    def _forward(self, method: str) -> None:
        path = urlsplit(self.path).path
        allowed = (method == "GET" and (path in GET_ROUTES or
                   path.startswith(("/api/demo/", "/api/gallery/", "/materials/")))) or \
                  (method == "POST" and path in POST_ROUTES)
        if not allowed:
            self.send_error(403, "Route not allowed by capture gateway")
            return
        payload = self.rfile.read(int(self.headers.get("Content-Length", "0"))) if method == "POST" else None
        headers = {"Content-Type": self.headers["Content-Type"]} if method == "POST" and self.headers.get("Content-Type") else {}
        request = Request(self.upstream + self.path, data=payload, headers=headers, method=method)
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            response = error
        except URLError:
            COUNTS[f"{method} upstream error"] += 1
            self.send_error(502, "Local API unavailable")
            return
        with response:
            body = response.read()
            self.send_response(response.status)
            for name in FORWARDED_HEADERS:
                value = response.headers.get(name)
                if value:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            COUNTS[f"{method} {path.split('/')[1]} {response.status}"] += 1

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.path = "/static/index.html"
            COUNTS["GET static index"] += 1
            return super().do_GET()
        if path.startswith("/static/"):
            requested = (self.static_root / unquote(path.removeprefix("/static/"))).resolve()
            if not requested.is_relative_to(self.static_root):
                self.send_error(403, "Static path outside current UI")
                return
            COUNTS["GET static asset"] += 1
            return super().do_GET()
        self._forward("GET")

    def do_POST(self) -> None:
        self._forward("POST")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", required=True, type=Path,
                        help="static/ directory in the integrated UI checkout")
    parser.add_argument("--api", required=True, help="existing real API on 127.0.0.1")
    parser.add_argument("--port", type=int, default=18073)
    args = parser.parse_args()
    target = urlsplit(args.api)
    if target.scheme != "http" or target.hostname not in {"127.0.0.1", "localhost"} or not target.port:
        parser.error("--api must be an explicit loopback HTTP address")
    static = args.static.resolve(strict=True)
    if not (static / "index.html").is_file():
        parser.error("--static must contain index.html")
    Gateway.upstream = args.api.rstrip("/")
    Gateway.static_root = static
    handler = partial(Gateway, directory=str(static.parent))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(json.dumps({"gateway": f"http://127.0.0.1:{args.port}",
                      "static": str(static), "api": args.api}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(json.dumps({"route_counts": dict(COUNTS)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
