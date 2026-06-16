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
    chunk_captions,
    clip_segments,
    export_captions,
    read_json,
    render_manifest_clip,
    title_card_seconds,
    title_card_text,
    words_in_range,
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
# How refined each render variant is. The most-refined variant per run is the
# one an editor usually wants (the final/revised cut), so it is flagged
# "preferred" and shown by default; earlier variants hide behind a toggle.
VARIANT_RANK = {"final": 3, "revise": 3, "draft": 1}


def variant_rank(variant: str) -> int:
    return VARIANT_RANK.get(variant.lower(), 2)

# Background single-clip render jobs and the previews they produce. Previews are
# served via /video even though they live under review/ (which discovery skips).
RENDER_JOBS: dict[str, dict[str, Any]] = {}
RENDER_JOBS_LOCK = threading.Lock()
RENDERED_PREVIEWS: set[str] = set()


def ffmpeg_available() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def load_env_file_minimal() -> None:
    """Load KEY=VALUE lines from the repo .env (for OPENAI_API_KEY etc.) without
    importing the heavy pipeline modules — keeps the core server dependency-free."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def openai_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("openai") is not None and bool(os.environ.get("OPENAI_API_KEY"))


REWORK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # strict json_schema requires every property to be listed in "required"
    "required": ["kept_sentences", "note"],
    "properties": {
        "kept_sentences": {"type": "array", "items": {"type": "integer"}},
        "note": {"type": "string"},
    },
}


def build_sentences(neigh: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group neighborhood words into sentences by terminal punctuation."""
    import re
    sents: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in neigh:
        cur.append(w)
        if re.search(r'[.!?]["\')]?$', w["text"]):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def rework_pricing():
    """Pricing for the editor's AI calls, read from .env (gpt-5.5 convention).

    Mirrors the per-stage *_USD_PER_MILLION vars; defaults to the project's
    standard gpt-5.5 numbers so a fresh checkout still reports a cost.
    """
    from extractor.openai_strategy import Pricing

    def num(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key) or default)
        except (TypeError, ValueError):
            return default

    return Pricing(
        input_usd_per_million=num("REVIEW_REWORK_INPUT_USD_PER_MILLION", 5.00),
        cached_input_usd_per_million=num("REVIEW_REWORK_CACHED_INPUT_USD_PER_MILLION", 0.50),
        output_usd_per_million=num("REVIEW_REWORK_OUTPUT_USD_PER_MILLION", 30.00),
    )


def response_cost(resp) -> dict[str, Any]:
    """Actual token usage + USD for a finished Responses call."""
    from extractor.openai_strategy import actual_cost, usage_dict

    response_dict = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {}
    return actual_cost(usage_dict(response_dict), rework_pricing())


def rework_clip(
    neigh: list[dict[str, Any]], in_idx: set[int], instruction: str
) -> tuple[set[int], str, dict[str, Any]]:
    """Ask the model to re-select sentences per the editor's instruction.

    `neigh` is the clip neighborhood (words with global index `gi`, text, start,
    end); `in_idx` is the currently-included global indices. Returns the new set
    of included global indices, a short note, and the call's actual cost. Lazy-
    imports openai so the core server has no third-party dependency.
    """
    from openai import OpenAI
    from extractor.openai_responses import create_response_streaming

    def ts(s: float) -> str:
        return f"{int(s // 60)}:{int(s % 60):02d}"

    sents = build_sentences(neigh)
    lines = []
    for i, s in enumerate(sents):
        inflag = any(w["gi"] in in_idx for w in s)
        text = " ".join(w["text"] for w in s)
        lines.append(f"[{i}]{'*' if inflag else ' '} ({ts(s[0]['start'])}-{ts(s[-1]['end'])}) {text}")

    developer = (
        "You are helping a video editor refine a short standalone excerpt from a "
        "lecture transcript. You are given numbered sentences from the source with "
        "timestamps; a leading '*' marks sentences currently IN the excerpt. Apply "
        "the editor's instruction and return kept_sentences: the indices that should "
        "be IN the revised excerpt, in source order. Keep it coherent and self-"
        "contained, starting and ending on clean sentence boundaries; do not invent "
        "indices outside the list."
    )
    user = f"Editor instruction: {instruction}\n\nSentences:\n" + "\n".join(lines)
    client = OpenAI(timeout=float(os.environ.get("OPENAI_CLIENT_TIMEOUT_SECONDS", "600")))
    resp = create_response_streaming(
        client, label="rework",
        model=os.environ.get("REVIEW_REWORK_MODEL") or "gpt-5.5",
        input=[{"role": "developer", "content": developer},
               {"role": "user", "content": user}],
        reasoning={"effort": os.environ.get("REVIEW_REWORK_EFFORT") or "low"},
        max_output_tokens=2000,
        store=False,
        text={"format": {"type": "json_schema", "name": "rework_selection",
                         "schema": REWORK_SCHEMA, "strict": True}},
    )
    out = json.loads(resp.output_text)
    kept = set(int(i) for i in out.get("kept_sentences", []))
    new_in: set[int] = set()
    for i, s in enumerate(sents):
        if i in kept:
            new_in.update(w["gi"] for w in s)
    return new_in, str(out.get("note", "")).strip(), response_cost(resp)


CAPTION_CLEANUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["captions", "summary"],
    "properties": {
        "captions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["i", "text"],
                "properties": {"i": {"type": "integer"}, "text": {"type": "string"}},
            },
        },
        "summary": {"type": "string"},
    },
}


def cleanup_captions(
    chunks: list[dict[str, Any]], instruction: str = ""
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Faithfully clean auto-generated (ASR) caption lines.

    `chunks` are source-anchored verbatim caption lines ({source_start,
    source_end, text}). The model fixes transcription errors, stutters, and
    punctuation WITHOUT paraphrasing, returning one cleaned line per index.
    Timing is preserved (only the text changes). Returns cleaned chunks, a
    one-sentence summary of the fixes, and the call's actual cost.
    """
    from openai import OpenAI
    from extractor.openai_responses import create_response_streaming

    lines = [f"[{i}] {c['text']}" for i, c in enumerate(chunks)]
    developer = (
        "You are cleaning up auto-generated (speech-to-text) captions for a short "
        "video clip. The text is verbatim and may contain transcription errors, "
        "false-start stutters (e.g. 'p- people', 'i-it'), missing or wrong "
        "punctuation, and bad capitalization. Fix ONLY those: correct obvious "
        "mis-transcriptions, remove stutters and false starts, and add natural "
        "punctuation and capitalization. Stay FAITHFUL to what was said — do NOT "
        "paraphrase, translate, summarize, reorder, or add or drop content words. "
        "Keep each line's wording and length close to the original (it must still "
        "match the spoken audio under it). Return exactly one cleaned line per "
        "input index in `captions`, plus a one-sentence `summary` of the kinds of "
        "fixes you made."
    )
    user = (f"Editor note: {instruction}\n\n" if instruction else "") + "Captions:\n" + "\n".join(lines)
    client = OpenAI(timeout=float(os.environ.get("OPENAI_CLIENT_TIMEOUT_SECONDS", "600")))
    resp = create_response_streaming(
        client, label="caption-cleanup",
        model=os.environ.get("REVIEW_CAPTION_MODEL") or os.environ.get("REVIEW_REWORK_MODEL") or "gpt-5.5",
        input=[{"role": "developer", "content": developer},
               {"role": "user", "content": user}],
        reasoning={"effort": os.environ.get("REVIEW_CAPTION_EFFORT") or "low"},
        max_output_tokens=int(os.environ.get("REVIEW_CAPTION_MAX_OUTPUT_TOKENS") or 6000),
        store=False,
        text={"format": {"type": "json_schema", "name": "caption_cleanup",
                         "schema": CAPTION_CLEANUP_SCHEMA, "strict": True}},
    )
    out = json.loads(resp.output_text)
    cleaned = {int(x["i"]): str(x.get("text", "")).strip() for x in out.get("captions", [])}
    result = [
        {"source_start": c["source_start"], "source_end": c["source_end"],
         "text": cleaned.get(i) or c["text"]}
        for i, c in enumerate(chunks)
    ]
    return result, str(out.get("summary", "")).strip(), response_cost(resp)


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
        # --auto-foreground-scale matches how the pipeline renders finals (a
        # per-clip vertical punch-in) so the preview render looks right AND the
        # editor's framing offset has a crop to act on (without a punch-in the
        # frame is full-width and the offset is a no-op).
        cmd = [sys.executable, "-m", "extractor.render_plan",
               "--source-video", str(source), "--plan-output", str(plan_path),
               "--output-dir", str(out_dir), "--output-label", "", "--clip-id", clip_id,
               "--auto-foreground-scale"]
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
        variant = manifest_path.parent.name
        projects.append({
            "id": project_id(manifest_path),
            "label": project_label(manifest_path, manifest),
            "manifest_path": str(manifest_path),
            "clip_count": len(manifest.get("clips", []) or []),
            "source_found": bool(source_path and source_path.exists()),
            "variant": variant,
            "run_root": str(run_root_of(manifest_path)),
            "preferred": True,  # refined below once all variants of a run are known
        })
        if source_path and source_path.exists():
            allowed.add(str(source_path.resolve()))
        for clip in manifest.get("clips", []) or []:
            for key in ("output_path", "subtitled_output_path"):
                p = clip.get(key)
                if p and Path(str(p)).exists():
                    allowed.add(str(Path(str(p)).resolve()))
    # Within each run, only the most-refined variant is "preferred" (shown by
    # default); the rest hide behind the editor's "show all variants" toggle.
    best_rank: dict[str, int] = {}
    for p in projects:
        best_rank[p["run_root"]] = max(best_rank.get(p["run_root"], -1), variant_rank(p["variant"]))
    for p in projects:
        p["preferred"] = variant_rank(p["variant"]) == best_rank[p["run_root"]]
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


def project_detail(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
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
            "title_card_text": title_card_text(clip),
            "title_card_seconds": card,
            "segments": [{"source_in": s["source_in"], "source_out": s["source_out"]} for s in segments],
            "framing_offset": float(clip.get("framing_offset") or 0.0),
            "preview_url": video_url(preview) if preview and Path(str(preview)).exists() else None,
            "output_path": str(preview) if preview else None,
            "uses_assets": has_assets,
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
            elif parsed.path == "/api/rework":
                self._api_rework()
            elif parsed.path == "/api/caption-cleanup":
                self._api_caption_cleanup()
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
            "ai": openai_available(),
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
        source = manifest.get("source_video")
        transcript_path = manifest.get("transcript")
        words = (
            transcript_words_from_data(read_json(Path(str(transcript_path))))
            if transcript_path and Path(str(transcript_path)).exists() else []
        )
        duration = probe_duration(Path(str(source))) if source and Path(str(source)).exists() else None

        out_dir = run_root_of(manifest_path) / "review" / manifest_path.parent.name
        out_path = out_dir / "plan-output.human.json"
        # Merge onto a prior human plan when one exists so a single-clip save
        # keeps edits already saved for the other clips (apply_edits leaves
        # clips absent from the overlay untouched); otherwise start from the
        # original plan.
        base_path = out_path if out_path.exists() else Path(str(plan_path))
        plan = read_json(base_path)
        revised = apply_edits(plan, {"clips": body.get("clips", [])}, words, duration,
                              source="clipinator-review-editor")
        out_dir.mkdir(parents=True, exist_ok=True)
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

    def _api_rework(self) -> None:
        if not openai_available():
            raise RuntimeError("AI rework needs the openai package and OPENAI_API_KEY in .env.")
        from urllib.parse import unquote
        body = self._read_json_body()
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            raise RuntimeError("Type an instruction for the AI (e.g. 'tighten the middle').")
        segs = body.get("segments", []) or []
        if not segs:
            raise RuntimeError("Select some words first so the AI knows the current cut.")
        manifest_path = Path(unquote(str(body.get("project_id", ""))))
        if not manifest_path.exists():
            raise RuntimeError("That project could not be found.")
        manifest = read_json(manifest_path)
        transcript_path = manifest.get("transcript")
        if not (transcript_path and Path(str(transcript_path)).exists()):
            raise RuntimeError("The transcript for this project is missing.")
        words = transcript_words_from_data(read_json(Path(str(transcript_path))))

        lo = min(s["source_in"] for s in segs) - 45.0
        hi = max(s["source_out"] for s in segs) + 45.0
        neigh = [
            {"gi": i, "text": w["text"], "start": float(w["start"]), "end": float(w["end"])}
            for i, w in enumerate(words)
            if float(w["end"]) >= lo and float(w["start"]) <= hi
        ]
        in_idx = {n["gi"] for n in neigh
                  if any(s["source_in"] - 1e-6 <= n["start"] <= s["source_out"] + 1e-6 for s in segs)}
        new_in, note, cost = rework_clip(neigh, in_idx, instruction)

        new_segs: list[dict[str, float]] = []
        run = None
        prev = None
        for i in sorted(new_in):
            w = words[i]
            if run is not None and prev == i - 1:
                run["source_out"] = float(w["end"])
            else:
                if run is not None:
                    new_segs.append(run)
                run = {"source_in": float(w["start"]), "source_out": float(w["end"])}
            prev = i
        if run is not None:
            new_segs.append(run)
        self._json({
            "segments": [{"source_in": round(s["source_in"], 3), "source_out": round(s["source_out"], 3)}
                         for s in new_segs],
            "note": note,
            "cost": cost,
        })

    def _api_caption_cleanup(self) -> None:
        if not openai_available():
            raise RuntimeError("Caption cleanup needs the openai package and OPENAI_API_KEY in .env.")
        from urllib.parse import unquote
        body = self._read_json_body()
        segs = body.get("segments", []) or []
        if not segs:
            raise RuntimeError("Select some words first so there are captions to clean up.")
        manifest_path = Path(unquote(str(body.get("project_id", ""))))
        if not manifest_path.exists():
            raise RuntimeError("That project could not be found.")
        manifest = read_json(manifest_path)
        transcript_path = manifest.get("transcript")
        if not (transcript_path and Path(str(transcript_path)).exists()):
            raise RuntimeError("The transcript for this project is missing.")
        words = transcript_words_from_data(read_json(Path(str(transcript_path))))

        # Build verbatim caption lines from the current cut, exactly as the save
        # path seeds them, so cleaned text lines up with what would render.
        segment_words: list[dict[str, Any]] = []
        for s in segs:
            segment_words.extend(words_in_range(words, float(s["source_in"]), float(s["source_out"])))
        chunks = chunk_captions(segment_words)
        if not chunks:
            raise RuntimeError("No captions to clean up on the current cut.")
        cleaned, summary, cost = cleanup_captions(chunks, str(body.get("instruction", "")).strip())
        changed = sum(1 for a, b in zip(chunks, cleaned) if a["text"] != b["text"])
        self._json({
            "captions": [
                {"source_start": round(c["source_start"], 3), "source_end": round(c["source_end"], 3),
                 "text": c["text"], "was": chunks[i]["text"]}
                for i, c in enumerate(cleaned)
            ],
            "changed": changed,
            "summary": summary,
            "cost": cost,
        })

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

    load_env_file_minimal()  # make OPENAI_API_KEY available for AI rework
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
