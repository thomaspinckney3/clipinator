from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from extractor.edit_list import (
    apply_edits,
    clip_segments,
    export_captions,
    read_json,
    render_manifest_clip,
    title_card_seconds,
    write_json,
)
from extractor.render_plan import transcript_words_from_data

# Local review server for the clipinator edit editor. Stdlib only (no third-
# party deps — see project-notes/editor-app/SPEC.md), so it runs under any
# Python 3.11+ via the .command launcher with PYTHONPATH=src. It discovers
# rendered runs in a library folder, range-serves the source video and clip
# previews for in-browser scrubbing, and exposes a small JSON API. ffmpeg is
# not needed here; it is only used when the (future) render endpoint shells out
# to render-plan.

CONFIG_DIR = Path.home() / ".clipinator-review"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_LIBRARY = Path.home() / "Documents" / "clipinator"
STATIC_DIR = Path(__file__).resolve().parent / "review_static"
SCAN_MAX_DEPTH = 7
SCAN_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "Library", ".Trash", "review"}
SCAN_MAX_PROJECTS = 500

# Background single-clip render jobs and the previews they produce. Previews are
# served via /video even though they live under review/ (which discovery skips).
RENDER_JOBS: dict[str, dict[str, Any]] = {}
RENDER_JOBS_LOCK = threading.Lock()
RENDERED_PREVIEWS: set[str] = set()


def ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def start_render_job(*, plan_path: Path, source: Path, transcript: Path | None,
                     out_dir: Path, clip_id: str) -> str:
    import sys
    import uuid
    job_id = uuid.uuid4().hex[:12]
    with RENDER_JOBS_LOCK:
        RENDER_JOBS[job_id] = {"status": "running", "message": "Rendering…"}

    def worker() -> None:
        import subprocess
        out_dir.mkdir(parents=True, exist_ok=True)
        # render_plan imports only stdlib + extractor; point PYTHONPATH at the
        # package root so the subprocess finds it whether run from src or installed.
        import extractor
        src_root = str(Path(extractor.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": src_root}
        cmd = [sys.executable, "-m", "extractor.render_plan",
               "--source-video", str(source), "--plan-output", str(plan_path),
               "--output-dir", str(out_dir), "--output-label", "", "--clip-id", clip_id]
        if transcript is not None:
            cmd += ["--transcript", str(transcript)]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-1:] or ["render failed"]
            with RENDER_JOBS_LOCK:
                RENDER_JOBS[job_id] = {"status": "error", "message": tail[0]}
            return
        mp4 = out_dir / f"{clip_id}.mp4"
        if not mp4.exists():
            with RENDER_JOBS_LOCK:
                RENDER_JOBS[job_id] = {"status": "error", "message": "Render finished but no file was produced."}
            return
        RENDERED_PREVIEWS.add(str(mp4.resolve()))
        with RENDER_JOBS_LOCK:
            RENDER_JOBS[job_id] = {"status": "done", "message": "Done.", "output": str(mp4)}

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"library": str(DEFAULT_LIBRARY)}


def save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def library_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("library") or DEFAULT_LIBRARY).expanduser()


def iter_render_manifests(root: Path) -> list[Path]:
    """Bounded walk for render-manifest.json files, skipping heavy/hidden dirs."""
    found: list[Path] = []
    root = root.expanduser()
    if not root.is_dir():
        return found
    root_depth = len(root.parts)
    # followlinks=True so a library of symlinks to run dirs on another drive
    # (e.g. big media on an external volume) is discovered; guard against cycles.
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= SCAN_MAX_DEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SCAN_SKIP_DIRS and not d.startswith(".")]
        if "render-manifest.json" in filenames:
            found.append(Path(dirpath) / "render-manifest.json")
            if len(found) >= SCAN_MAX_PROJECTS:
                break
    return found


def project_id(manifest_path: Path) -> str:
    return quote(str(manifest_path), safe="")


def project_label(manifest_path: Path, manifest: dict[str, Any]) -> str:
    # The run root is the ancestor that contains "renders":
    # <run-root>/renders/[<mode>/]<variant>/render-manifest.json. Use the run
    # root's folder name (clean, editor-chosen) regardless of where we scanned.
    parts = manifest_path.parts
    name = manifest_path.parent.parent.name
    if "renders" in parts:
        idx = len(parts) - 1 - parts[::-1].index("renders")
        if idx >= 1:
            name = parts[idx - 1]
    variant = manifest_path.parent.name  # draft / final / human / ...
    return f"{name} — {variant}"


def discover_projects(root: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Return (projects, allowed_media_paths). Media allowlist gates /video."""
    projects: list[dict[str, Any]] = []
    allowed: set[str] = set()
    for manifest_path in iter_render_manifests(root):
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        source = manifest.get("source_video")
        source_path = Path(str(source)) if source else None
        projects.append({
            "id": project_id(manifest_path),
            "label": project_label(manifest_path, manifest),
            "manifest_path": str(manifest_path),
            "clip_count": len(manifest.get("clips", []) or []),
            "source_found": bool(source_path and source_path.exists()),
        })
        if source_path and source_path.exists():
            allowed.add(str(source_path.resolve()))
        for clip in manifest.get("clips", []) or []:
            for key in ("output_path", "subtitled_output_path"):
                p = clip.get(key)
                if p and Path(str(p)).exists():
                    allowed.add(str(Path(str(p)).resolve()))
    projects.sort(key=lambda p: p["label"].lower())
    return projects, allowed


def video_url(path: str | None) -> str | None:
    if not path:
        return None
    return "/video?path=" + quote(str(path), safe="")


def run_root_of(manifest_path: Path) -> Path:
    """The run root is the ancestor containing 'renders'."""
    parts = manifest_path.parts
    if "renders" in parts:
        idx = len(parts) - 1 - parts[::-1].index("renders")
        return Path(*parts[:idx])
    return manifest_path.parent.parent


def probe_duration(video: Path) -> float | None:
    """Source duration via ffprobe, or None if ffprobe is absent/fails (ffmpeg
    is optional; without it, import skips the end-of-video bounds check)."""
    import shutil
    import subprocess
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def clip_words(words: list[dict[str, Any]], segments: list[dict[str, Any]], pad: float = 4.0) -> list[dict[str, Any]]:
    """Transcript words spanning the clip (plus padding), each flagged in/out.

    'in' = the word falls within one of the current segments. Padding words
    (initially out) let the editor extend the clip. This is the transcript-first
    edit surface: toggling words re-derives the segments on save.
    """
    if not segments or not words:
        return []
    lo = min(s["source_in"] for s in segments) - pad
    hi = max(s["source_out"] for s in segments) + pad
    out: list[dict[str, Any]] = []
    for w in words:
        start, end = float(w["start"]), float(w["end"])
        if end < lo or start > hi:
            continue
        inside = any(s["source_in"] - 1e-6 <= start <= s["source_out"] + 1e-6 for s in segments)
        out.append({"text": str(w.get("text", "")), "start": round(start, 3),
                    "end": round(end, 3), "in": inside})
    return out


def load_critique(run_root: Path) -> dict[str, dict[str, Any]]:
    """Map clip_id -> critique review from the run's critique-output.json, if any.

    Surfaces the AI's existing per-clip review (verdict + issues) so the editor
    can see what the model flagged. No API call.
    """
    reviews: dict[str, dict[str, Any]] = {}
    runs = run_root / "openai-runs"
    if not runs.is_dir():
        return reviews
    for cand in sorted(runs.glob("*critique*/critique-output.json")):
        try:
            data = read_json(cand)
        except (OSError, json.JSONDecodeError):
            continue
        for r in data.get("clip_reviews", []) or []:
            cid = str(r.get("clip_id") or "")
            if cid:
                reviews.setdefault(cid, r)
        break  # first critique run is enough
    return reviews


def clip_critique(reviews: dict[str, dict[str, Any]], clip_id: str) -> dict[str, Any] | None:
    """Match a clip to its review by exact id, else by prefix (final clips carry
    a revise suffix like '_R2' over the draft critique's clip_id)."""
    if clip_id in reviews:
        return reviews[clip_id]
    for cid, r in reviews.items():
        if clip_id.startswith(cid) or cid.startswith(clip_id):
            return r
    return None


def compact_critique(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if not review:
        return None
    return {
        "verdict": review.get("verdict"),
        "score": review.get("score_0_to_100"),
        "assessment": review.get("one_sentence_assessment"),
        "issues": [
            {"category": i.get("category"), "severity": i.get("severity"),
             "why": i.get("why_it_matters") or i.get("evidence"),
             "fix": i.get("suggested_fix")}
            for i in (review.get("issues") or [])
        ],
    }


def project_detail(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    reviews = load_critique(run_root_of(manifest_path))
    source = manifest.get("source_video")
    plan_path = manifest.get("stage3_output")  # legacy key retained in manifests
    transcript_path = manifest.get("transcript")
    words: list[dict[str, Any]] = []
    if transcript_path and Path(str(transcript_path)).exists():
        words = transcript_words_from_data(read_json(Path(str(transcript_path))))
    plan = read_json(Path(str(plan_path))) if plan_path and Path(str(plan_path)).exists() else {}

    clips: list[dict[str, Any]] = []
    for clip in plan.get("clip_ads", []) or []:
        clip_id = str(clip.get("clip_id", ""))
        segments = clip_segments(clip)
        card = title_card_seconds(clip)
        manifest_clip = render_manifest_clip(manifest, clip_id)
        preview = manifest_clip.get("output_path") if manifest_clip else None
        has_assets = bool(clip.get("media_request_ids") or any(
            "asset://" in str(item.get("visual_source", "")) for item in clip.get("timeline", []) or []
        ))
        clips.append({
            "clip_id": clip_id,
            "title": clip.get("concept_title", ""),
            "title_card_seconds": card,
            "segments": [{"source_in": s["source_in"], "source_out": s["source_out"]} for s in segments],
            "preview_url": video_url(preview) if preview and Path(str(preview)).exists() else None,
            "uses_assets": has_assets,
            "critique": compact_critique(clip_critique(reviews, clip_id)),
        })
    return {
        "id": project_id(manifest_path),
        "label": project_label(manifest_path, manifest),
        "content_mode": manifest.get("content_mode", "ad"),
        "source_url": video_url(source) if source and Path(str(source)).exists() else None,
        "source_found": bool(source and Path(str(source)).exists()),
        "transcript_word_count": len(words),
        # the full transcript, so the editor can reveal and include content before
        # or after where a clip currently starts/ends (front-end windows it)
        "transcript_words": [
            {"text": str(w.get("text", "")), "start": round(float(w["start"]), 3),
             "end": round(float(w["end"]), 3)}
            for w in words
        ],
        "clips": clips,
    }


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ClipinatorReview/0.1"

    # injected by make_server
    allowed_media: set[str] = set()

    def log_message(self, *args: Any) -> None:  # quiet; this is a desktop tool
        pass

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status=status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route == "/" or route == "/index.html":
                self._serve_static("index.html")
            elif route.startswith("/static/"):
                self._serve_static(route[len("/static/"):])
            elif route == "/api/projects":
                self._api_projects()
            elif route == "/api/project":
                self._api_project(query.get("id", [""])[0])
            elif route == "/api/config":
                self._json(load_config())
            elif route == "/api/render-status":
                self._api_render_status(query.get("job", [""])[0])
            elif route == "/video":
                self._serve_video(query.get("path", [""])[0])
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found.")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client closed the connection (common while scrubbing video)
        except Exception as exc:  # never leak a traceback to the browser
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Something went wrong: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/save":
                self._api_save()
            elif parsed.path == "/api/render":
                self._api_render()
            else:
                self._error(HTTPStatus.NOT_FOUND, "Not found.")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except RuntimeError as exc:  # validation / known errors -> plain message
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Could not save: {exc}")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    # --- routes -----------------------------------------------------------

    def _api_projects(self) -> None:
        config = load_config()
        projects, allowed = discover_projects(library_dir(config))
        type(self).allowed_media = allowed
        self._json({
            "library": str(library_dir(config)),
            "projects": projects,
            "ffmpeg": ffmpeg_available(),
        })

    def _api_project(self, pid: str) -> None:
        from urllib.parse import unquote
        manifest_path = Path(unquote(pid))
        if not manifest_path.exists():
            self._error(HTTPStatus.NOT_FOUND, "That project could not be found.")
            return
        # refresh allowlist so this project's media is serveable
        _, allowed = discover_projects(library_dir(load_config()))
        type(self).allowed_media = allowed
        self._json(project_detail(manifest_path))

    def _write_human_plan(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply the edit overlay from the browser and write the human plan.

        Shared by save and render. Returns paths needed to render afterward.
        """
        from urllib.parse import unquote
        manifest_path = Path(unquote(str(body.get("project_id", ""))))
        if not manifest_path.exists():
            raise RuntimeError("That project could not be found.")
        manifest = read_json(manifest_path)
        plan_path = manifest.get("stage3_output")
        if not (plan_path and Path(str(plan_path)).exists()):
            raise RuntimeError("The plan for this project is missing.")
        plan = read_json(Path(str(plan_path)))
        source = manifest.get("source_video")
        transcript_path = manifest.get("transcript")
        words = (
            transcript_words_from_data(read_json(Path(str(transcript_path))))
            if transcript_path and Path(str(transcript_path)).exists() else []
        )
        duration = probe_duration(Path(str(source))) if source and Path(str(source)).exists() else None

        revised = apply_edits(plan, {"clips": body.get("clips", [])}, words, duration,
                              source="clipinator-review-editor")
        out_dir = run_root_of(manifest_path) / "review" / manifest_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "plan-output.human.json"
        write_json(out_path, revised)
        return {
            "revised": revised, "out_path": out_path, "out_dir": out_dir,
            "source": source, "transcript": transcript_path,
        }

    def _api_save(self) -> None:
        info = self._write_human_plan(self._read_json_body())
        he = info["revised"]["human_edit"]
        self._json({
            "ok": True,
            "output": str(info["out_path"]),
            "kept": len(info["revised"]["clip_ads"]),
            "dropped": he["dropped_clip_ids"],
            "reworked": he["rework_clip_ids"],
        })

    def _api_render(self) -> None:
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg is not installed; install it to render clips here.")
        body = self._read_json_body()
        clip_id = str(body.get("clip_id", ""))
        if not clip_id:
            raise RuntimeError("No clip specified to render.")
        info = self._write_human_plan(body)  # save current edits first
        if not (info["source"] and Path(str(info["source"])).exists()):
            raise RuntimeError("The source video for this project was not found.")
        job_id = start_render_job(
            plan_path=info["out_path"],
            source=Path(str(info["source"])),
            transcript=Path(str(info["transcript"])) if info["transcript"] else None,
            out_dir=info["out_dir"] / "preview-renders",
            clip_id=clip_id,
        )
        self._json({"job_id": job_id})

    def _api_render_status(self, job_id: str) -> None:
        job = RENDER_JOBS.get(job_id)
        if not job:
            self._error(HTTPStatus.NOT_FOUND, "Unknown render job.")
            return
        payload = {"status": job["status"], "message": job.get("message", "")}
        if job["status"] == "done" and job.get("output"):
            payload["output_url"] = video_url(job["output"])
        self._json(payload)

    def _serve_static(self, rel: str) -> None:
        path = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve() / rel:
            self._error(HTTPStatus.FORBIDDEN, "Forbidden.")
            return
        if not path.exists() or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "Not found.")
            return
        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_video(self, raw_path: str) -> None:
        if not raw_path:
            self._error(HTTPStatus.BAD_REQUEST, "Missing path.")
            return
        path = Path(raw_path)
        if str(path.resolve()) not in type(self).allowed_media and str(path.resolve()) not in RENDERED_PREVIEWS:
            self._error(HTTPStatus.FORBIDDEN, "That file is not part of a known project.")
            return
        if not path.exists():
            self._error(HTTPStatus.NOT_FOUND, "Video file not found.")
            return
        self._serve_file_with_ranges(path)

    def _serve_file_with_ranges(self, path: Path) -> None:
        file_size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "video/mp4"
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes="):].split(",")[0].strip()
            lo, _, hi = spec.partition("-")
            try:
                if lo:
                    start = int(lo)
                    end = int(hi) if hi else file_size - 1
                elif hi:  # suffix range: last N bytes
                    start = max(0, file_size - int(hi))
            except ValueError:
                start, end = 0, file_size - 1
            start = max(0, start)
            end = min(end, file_size - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser aborted the range request (normal while scrubbing)
                remaining -= len(chunk)


def find_free_port(preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local clipinator review editor server.")
    parser.add_argument(
        "target", nargs="?", type=Path, default=None,
        help="A project run folder, or a folder of projects, to open. If omitted, "
             "reopens the last folder (or ~/Documents/clipinator).",
    )
    parser.add_argument("--library", type=Path, default=None, help="Same as the positional folder.")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    chosen = args.target or args.library
    if chosen:
        # Point at a single project run root or a folder of projects; the scan
        # handles both. Remember it as the last-used folder.
        config["library"] = str(chosen.expanduser().resolve())
        save_config(config)
    elif not CONFIG_PATH.exists():
        save_config(config)

    port = args.port or find_free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Clipinator review editor running at {url}")
    print(f"Library folder: {library_dir(config)}")
    print("Close this window (or press Ctrl-C) to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
