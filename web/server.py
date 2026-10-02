"""Web UI for the hiring agent. Serves the page and scores an uploaded PDF."""

import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# providers.json defaults to a local Ollama model. On Vercel, use OpenAI
# unless DEFAULT_MODEL is already set in the project env.
if os.environ.get("VERCEL") and not os.environ.get("DEFAULT_MODEL"):
    os.environ["DEFAULT_MODEL"] = "gpt-4.1"

import github as github_mod
import score as score_mod
from roles import list_available_roles, load_role

# Don't store uploaded resumes or evaluation rows. Do reuse public GitHub cache
# so a repeat score doesn't burn the unauthenticated rate limit and stall.
score_mod.DEVELOPMENT_MODE = False
# Vercel can only write to /tmp, so skip the GitHub file cache there.
github_mod.DEVELOPMENT_MODE = not os.environ.get("VERCEL")

WEB = ROOT / "public"
MAX_BYTES = 8 * 1024 * 1024
SCORE_LOCK = threading.Lock()
ROLES = {name: load_role(name) for name in list_available_roles()}


def parse_multipart(content_type, body):
    if "multipart/form-data" not in (content_type or ""):
        raise ValueError("Upload the resume as a form.")
    boundary = None
    for piece in content_type.split(";"):
        piece = piece.strip()
        if piece.startswith("boundary="):
            boundary = piece.split("=", 1)[1].strip().strip('"')
    if not boundary:
        raise ValueError("Upload was missing a form boundary.")

    fields = {}
    files = {}
    delimiter = b"--" + boundary.encode()
    for chunk in body.split(delimiter):
        if chunk in (b"", b"--", b"--\r\n"):
            continue
        chunk = chunk.strip(b"\r\n")
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].rstrip(b"\r\n")
        if b"\r\n\r\n" not in chunk:
            continue
        raw_head, data = chunk.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        headers = {}
        for line in raw_head.decode("utf-8", "replace").split("\r\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
        name = filename = None
        for piece in headers.get("content-disposition", "").split(";"):
            piece = piece.strip()
            if piece.startswith("name="):
                name = piece.split("=", 1)[1].strip().strip('"')
            elif piece.startswith("filename="):
                filename = piece.split("=", 1)[1].strip().strip('"')
        if not name:
            continue
        if filename is None:
            fields[name] = data.decode("utf-8", "replace")
        else:
            files[name] = {
                "filename": filename,
                "type": headers.get("content-type", ""),
                "data": data,
            }
    return fields, files


def public_error(exc):
    text = str(exc)
    if "401" in text or "Unauthorized" in text:
        return (
            "The PDF was received. Scoring never started because the model rejected "
            "the API key in .env. Put a working key there and upload again."
        )
    if "429" in text:
        return "The model is rate-limiting requests. Wait a minute and try again."
    if "Unknown model" in text or "requires" in text and "env var" in text:
        return text
    if not text:
        return "Scoring failed."
    return text if len(text) <= 500 else text[:500]


def report(evaluation, role, candidate_name):
    if not evaluation:
            raise RuntimeError(
                "Scoring stopped before a report was produced. "
                "Check the model key in .env, or upload a text-based PDF resume."
            )

    total = 0.0
    out_of = 0
    categories = []
    dumped = evaluation.scores.model_dump()
    for category in role.categories:
        data = dumped.get(category.key)
        if not data:
            continue
        capped = min(float(data["score"]), float(data["max"]))
        total += capped
        out_of += int(data["max"])
        categories.append(
            {
                "key": category.key,
                "label": category.label,
                "score": capped,
                "max": int(data["max"]),
                "evidence": data["evidence"],
            }
        )

    bonus = evaluation.bonus_points
    deductions = evaluation.deductions
    if bonus:
        total += float(bonus.total)
    if deductions:
        total -= float(deductions.total)
    ceiling = out_of + role.bonus_max
    if total > ceiling:
        total = float(ceiling)

    return {
        "candidate": candidate_name or "Candidate",
        "role": role.name,
        "position": role.position_title,
        "score": total,
        "out_of": out_of,
        "categories": categories,
        "bonus": (
            {"total": float(bonus.total), "breakdown": bonus.breakdown}
            if bonus
            else None
        ),
        "deductions": (
            {"total": float(deductions.total), "reasons": deductions.reasons}
            if deductions
            else None
        ),
        "strengths": list(evaluation.key_strengths or []),
        "gaps": list(evaluation.areas_for_improvement or []),
    }


def score_upload(data, filename, role_name):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=True)
    if role_name not in ROLES:
        raise ValueError("Pick a role that exists.")
    if not data:
        raise ValueError("That file is empty.")
    if not data.startswith(b"%PDF"):
        raise ValueError("Upload a PDF resume.")
    suffix = ".pdf"
    if filename and not filename.lower().endswith(".pdf"):
        raise ValueError("Upload a PDF resume.")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        with SCORE_LOCK:
            evaluation, candidate_name = score_mod.main(tmp.name, ROLES[role_name])
        return report(evaluation, ROLES[role_name], candidate_name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def match_upload(data, filename, job_text):
    """Score one resume against a pasted job description."""
    from dotenv import load_dotenv

    from llm_utils import extract_json_from_response, initialize_llm_provider
    from pdf import PDFHandler
    from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

    load_dotenv(ROOT / ".env", override=True)
    job_text = " ".join((job_text or "").split())
    if len(job_text) < 40:
        raise ValueError("Paste a job description. A title alone is not enough.")
    if len(job_text) > 12000:
        job_text = job_text[:12000]
    if not data or not data.startswith(b"%PDF"):
        raise ValueError("Upload a PDF resume.")
    if filename and not filename.lower().endswith(".pdf"):
        raise ValueError("Upload a PDF resume.")

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        with SCORE_LOCK:
            resume = PDFHandler().extract_json_from_pdf(tmp.name)
        if resume is None:
            raise RuntimeError(
                "Scoring stopped before a report was produced. "
                "Check the model key in .env, or upload a text-based PDF resume."
            )
        dumped = resume.model_dump()
        basics = dumped.get("basics") or {}
        basics["email"] = None
        basics["phone"] = None
        name = basics.get("name") or "Candidate"
        profile = json.dumps(dumped, ensure_ascii=False)[:24000]
        params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9})
        provider = initialize_llm_provider(DEFAULT_MODEL)
        response = provider.chat(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Compare a resume to one job description. Reply with JSON only. "
                        "Score fit from 0 to 100 using only facts in the resume. "
                        "Do not invent tools, employers, or numbers. "
                        "Name, college, grades, and city must not change the score. "
                        "Keys: candidate (string), score (number), summary (one sentence), "
                        "matched (list of requirements the resume meets), "
                        "missing (list of requirements it does not show)."
                    ),
                },
                {
                    "role": "user",
                    "content": f"JOB DESCRIPTION:\n{job_text}\n\nRESUME JSON:\n{profile}",
                },
            ],
            options={
                "temperature": params.get("temperature", 0.1),
                "top_p": params.get("top_p", 0.9),
            },
            format={"type": "object"},
        )
        parsed = json.loads(extract_json_from_response(response["message"]["content"]))
        score = max(0.0, min(100.0, float(parsed.get("score") or 0)))
        return {
            "candidate": parsed.get("candidate") or name,
            "score": score,
            "out_of": 100,
            "summary": str(parsed.get("summary") or ""),
            "matched": [str(item) for item in (parsed.get("matched") or [])][:8],
            "missing": [str(item) for item in (parsed.get("missing") or [])][:8],
        }
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def job_request(body, content_type):
    if not body or len(body) > MAX_BYTES + 64 * 1024:
        return 400, {"error": "Upload a PDF under 8 MB."}
    try:
        fields, files = parse_multipart(content_type, body)
        upload = files.get("resume")
        if not upload:
            raise ValueError("Choose a PDF resume.")
        if len(upload["data"]) > MAX_BYTES:
            raise ValueError("Upload a PDF under 8 MB.")
        return 200, match_upload(upload["data"], upload["filename"], fields.get("job"))
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except RuntimeError as exc:
        return 502, {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        return 502, {"error": public_error(exc)}


def roles_payload():
    return {
        "roles": [
            {
                "name": role.name,
                "position": role.position_title,
                "bonus_max": role.bonus_max,
                "categories": [
                    {
                        "key": category.key,
                        "label": category.label,
                        "max": category.max,
                    }
                    for category in role.categories
                ],
            }
            for role in ROLES.values()
        ]
    }


def score_request(body, content_type):
    """Score one upload. Returns an HTTP status and a JSON-ready dict."""
    if not body or len(body) > MAX_BYTES + 64 * 1024:
        return 400, {"error": "Upload a PDF under 8 MB."}
    try:
        fields, files = parse_multipart(content_type, body)
        upload = files.get("resume")
        if not upload:
            raise ValueError("Choose a PDF resume.")
        if len(upload["data"]) > MAX_BYTES:
            raise ValueError("Upload a PDF under 8 MB.")
        role_name = fields.get("role") or next(iter(ROLES))
        return 200, score_upload(upload["data"], upload["filename"], role_name)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    except RuntimeError as exc:
        return 502, {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        return 502, {"error": public_error(exc)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status, body, content_type):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status, data):
        self._send(status, json.dumps(data), "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        path = path.rstrip("/") or "/"
        pages = {
            "/": "index.html",
            "/example": "example.html",
            "/algorithm": "algorithm.html",
            "/faq": "faq.html",
            "/job": "job.html",
        }
        if path in pages:
            self._send(200, (WEB / pages[path]).read_bytes(), "text/html; charset=utf-8")
            return
        name = path[1:]
        file = WEB / name
        types = {
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }
        if "/" not in name and file.is_file() and file.suffix in types:
            self._send(200, file.read_bytes(), types[file.suffix])
            return
        if path == "/favicon.ico" and (WEB / "favicon.svg").is_file():
            self._send(200, (WEB / "favicon.svg").read_bytes(), "image/svg+xml")
            return
        if path == "/api/roles":
            self._json(200, roles_payload())
            return
        self._json(404, {"error": "Not found."})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/score", "/api/job"):
            self._json(404, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        content_type = self.headers.get("Content-Type", "")
        if path == "/api/job":
            status, payload = job_request(body, content_type)
        else:
            status, payload = score_request(body, content_type)
        self._json(status, payload)


handler = Handler


def main():
    if not ROLES:
        sys.exit("No roles found under roles/.")
    port = int(os.environ.get("PORT", "8787"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Hiring agent UI at http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
