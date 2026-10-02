"""Vercel function: GET /api/roles"""

import json
from http.server import BaseHTTPRequestHandler

from web.server import roles_payload


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps(roles_payload()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)
