"""Re-key stage response caches so a relocated project's existing model responses
become reachable by a future pipeline run.

Background. The local response cache is content-addressed: a stage hits it only
when it recomputes the exact key an entry is stored under. Two things move that
key out from under an old entry:
  1. filesystem paths that used to be baked into the prompt (fixed forward by
     stage_cache.without_location), and
  2. the source-file `implementation_hash` in "strict" mode, which any refactor
     bumps.
So a project migrated to the source-centric layout finds its months-old responses
unreachable even though the actual question (transcript + media facts) is unchanged.

This tool revives those responses without re-calling the API. For each stage run
dir it:
  - re-renders the CURRENT prompt and diffs it against the run's stored
    `source-materials.txt`; it proceeds ONLY if every difference is a filesystem
    location line (path/filename/stem) — i.e. the model's actual question is
    unchanged, so reusing the old answer is valid, not stale;
  - asks the stage's own `--dry-run` for the key the current code will look up
    (in query-only mode by default, so the revived entry also survives the next
    refactor); and
  - writes the run's stored response under that key.

Scope: strategy, moments, plan — the stages whose only location coupling is the
manifest/transcript paths this repo now strips. critique/revise embed render-output
paths that are not yet decoupled, so they are skipped (reported, not revived).

Because the gate fails closed, an imperfect re-render can only cause an
over-conservative skip, never a wrong revival. Default is a dry run; pass --apply
to write. Stdlib + the stage modules only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from extractor import openai_moments, openai_plan, openai_strategy, stage_cache

LOCATION_LINE = re.compile(r'^\s*"(path|filename|stem)"\s*:')
KEY_LINE = re.compile(r"local response cache key:\s*([0-9a-f]{64})")
CACHE_STATUS_LINE = re.compile(r"local response cache:\s*(hit|miss)")


def content_mode_of(run_dir: Path) -> str:
    return "excerpt" if "-excerpt-" in run_dir.name else "ad"


def _visual_index_section(module: Any, manifest_path: Path, max_images: int, label: str) -> str:
    """Reproduce the moments/plan visual-index append verbatim so the gate diff is
    faithful. `label` is the one line of instruction text that differs between the
    two stages."""
    visual_index = module.compact_visual_index_manifest(manifest_path=manifest_path, max_images=max_images)
    return "\n\n".join([
        "DENSE CANDIDATE VISUAL INDEX",
        label,
        "COMPACT VISUAL INDEX JSON:",
        json.dumps(visual_index, ensure_ascii=False, indent=2),
    ])


MOMENTS_VI_LABEL = (
    "Candidate visual-index contact sheets are attached after the main contact sheet "
    "in the same order as candidate_sheets below. Use these 1 fps sheets to verify "
    "concrete objects/actions when selecting candidate windows."
)
PLAN_VI_LABEL = (
    "Candidate visual-index contact sheets are attached after the main contact sheet "
    "in the same order as candidate_sheets below. Use these 1 fps sheets to verify "
    "concrete objects/actions before creating a render plan."
)


def render_strategy(paths: dict[str, Any], mi: int) -> str:
    return openai_strategy.build_source_materials(
        Path(paths["manifest"]), Path(paths["transcript"]),
        Path(paths["contact_sheet"]) if paths.get("contact_sheet") else None)


def render_moments(paths: dict[str, Any], mi: int) -> str:
    sm = render_strategy(paths, mi)
    if paths.get("visual_index_manifest"):
        sm = sm + "\n\n" + _visual_index_section(
            openai_moments, Path(paths["visual_index_manifest"]), mi, MOMENTS_VI_LABEL)
    return sm


def render_plan(paths: dict[str, Any], mi: int) -> str:
    sm = render_strategy(paths, mi)
    if paths.get("visual_index_manifest"):
        sm = sm + "\n\n" + _visual_index_section(
            openai_plan, Path(paths["visual_index_manifest"]), mi, PLAN_VI_LABEL)
    # NOTE: an augmentation-asset catalog can follow; if present we cannot fully
    # reproduce it here, so the gate diff will flag it and we skip (fail closed).
    return sm


# stage -> (render fn, extra dry-run argv builder from request-manifest)
def _argv_strategy(p: dict, run_dir: Path) -> list[str]:
    return ["--transcript", p["transcript"], "--manifest", p["manifest"],
            *(["--contact-sheet", p["contact_sheet"]] if p.get("contact_sheet") else [])]


def _argv_moments(p: dict, run_dir: Path) -> list[str]:
    return _argv_strategy(p, run_dir) + [
        "--strategy-output", p["strategy_output"],
        *(["--visual-index-manifest", p["visual_index_manifest"]] if p.get("visual_index_manifest") else [])]


def _argv_plan(p: dict, run_dir: Path) -> list[str]:
    return _argv_moments(p, run_dir) + [
        "--moments-output", p["moments_output"],
        *(["--augmentation-assets", p["augmentation_assets"]] if p.get("augmentation_assets") else [])]


STAGES: dict[str, dict[str, Any]] = {
    "strategy": {"module": "extractor.openai_strategy", "render": render_strategy, "argv": _argv_strategy},
    "moments": {"module": "extractor.openai_moments", "render": render_moments, "argv": _argv_moments},
    "plan": {"module": "extractor.openai_plan", "render": render_plan, "argv": _argv_plan},
}
UNSUPPORTED = ("critique", "revise")  # render-output-path coupling not yet decoupled


def stage_of(run_dir: Path) -> str | None:
    for s in list(STAGES) + list(UNSUPPORTED):
        if run_dir.name.endswith(f"-{s}"):
            return s
    return None


VIDEO_OPEN = re.compile(r'^(\s*)"video"\s*:\s*\{\s*$')
CLOSE = re.compile(r'^(\s*)\},?\s*$')


def strip_location(text: str) -> str:
    """Remove exactly the fields this repo drops from the fingerprint: any `video`
    object that carries a filesystem path (the manifest and visual-index location
    blocks — but NOT `media.video`, which is codec/dimension content), plus stray
    path/filename/stem lines (e.g. the transcript source). Applied to BOTH the
    stored June prompt and the current one so the two become comparable; whatever
    still differs afterward is a genuine content change."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = VIDEO_OPEN.match(lines[i])
        if m:
            indent = m.group(1)
            j = i + 1
            while j < len(lines) and not (CLOSE.match(lines[j]) and CLOSE.match(lines[j]).group(1) == indent):
                j += 1
            block = lines[i:j + 1]
            if any(LOCATION_LINE.match(b) for b in block):
                i = j + 1  # drop the whole location video block
                continue
        if LOCATION_LINE.match(lines[i]):
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def gate_paths_only(current: str, stored: str) -> tuple[bool, list[str]]:
    """True iff, after removing the location fields this repo strips from BOTH
    prompts, nothing differs — i.e. the model's actual question (transcript + media
    facts) is unchanged and reusing the old answer is valid. Fails closed: any
    residual difference (a template/content change, or a section this tool did not
    reproduce) is reported and the entry is skipped."""
    a, b = strip_location(stored).splitlines(), strip_location(current).splitlines()
    only = (Counter(a) - Counter(b)) + (Counter(b) - Counter(a))
    return (not only), sorted(l.strip() for l in only)[:8]


def dry_run_key(module: str, argv: list[str], cache_dir: Path, cache_mode: str) -> tuple[str | None, str | None]:
    """Run the stage's own dry-run (no API) and read the cache key + hit/miss it
    reports. This is the authoritative key a future run will look up."""
    cmd = [sys.executable, "-m", module, *argv, "--output-dir", str(_TMP.name),
           "--cache-dir", str(cache_dir), "--cache-mode", cache_mode, "--dry-run"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    key = KEY_LINE.search(out)
    status = CACHE_STATUS_LINE.search(out)
    if key is None:
        raise RuntimeError(f"no cache key reported ({module}):\n{out[-800:]}")
    return key.group(1), (status.group(1) if status else None)


def write_entry(run_dir: Path, cache_dir: Path, key: str, cache_mode: str) -> Path:
    """Copy the run's stored response artifacts under the new key."""
    entry = stage_cache.cache_entry_dir(cache_dir, key)
    entry.mkdir(parents=True, exist_ok=True)
    for name in ("raw-response.json", "output-text.json", "usage-and-cost.json"):
        src = run_dir / name
        if src.exists():
            (entry / name).write_bytes(src.read_bytes())
    # the parsed stage output (strategy-output.json / moments-output.json / ...)
    for out in run_dir.glob("*-output.json"):
        (entry / out.name).write_bytes(out.read_bytes())
    manifest = {
        "schema_version": 2,
        "cache_key": key,
        "cache_mode": cache_mode,
        "source": f"rekeyed_from_run:{run_dir}",
        "note": "Revived from a relocated run whose prompt differed only in file paths.",
    }
    (entry / "cache-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


def process_run(run_root: Path, cache_mode: str, apply: bool) -> list[dict[str, Any]]:
    cache_dir = run_root / "openai-runs" / "_response-cache"
    results: list[dict[str, Any]] = []
    for run_dir in sorted((run_root / "openai-runs").glob("*")):
        if not (run_dir / "request-manifest.json").exists():
            continue
        stage = stage_of(run_dir)
        if stage is None:
            continue
        rec: dict[str, Any] = {"stage": stage, "run_dir": run_dir.name}
        if stage in UNSUPPORTED:
            rec["status"] = "skip:unsupported-stage (render-path coupling)"
            results.append(rec); continue
        rm = json.loads((run_dir / "request-manifest.json").read_text())
        p = rm.get("paths") or {}
        stored_sm = run_dir / "source-materials.txt"
        if not stored_sm.exists() or not (run_dir / "raw-response.json").exists():
            rec["status"] = "skip:missing stored prompt or response"
            results.append(rec); continue
        spec = STAGES[stage]
        try:
            mi = int(rm.get("visual_index_max_images") or 24)
            current = spec["render"](p, mi)
        except Exception as exc:
            rec["status"] = f"skip:render-failed ({type(exc).__name__}: {exc})"
            results.append(rec); continue
        ok, offenders = gate_paths_only(current, stored_sm.read_text())
        if not ok:
            rec["status"] = "skip:prompt changed beyond paths"
            rec["offending"] = offenders
            results.append(rec); continue
        try:
            argv = spec["argv"](p, run_dir) + [
                "--content-mode", content_mode_of(run_dir),
                "--model", str(rm.get("model")), "--reasoning-effort", str(rm.get("reasoning_effort")),
                "--max-output-tokens", str(rm.get("max_output_tokens") or 24000)]
        except KeyError as exc:
            rec["status"] = f"skip:manifest missing input {exc}"
            results.append(rec); continue
        try:
            key, status = dry_run_key(spec["module"], argv, cache_dir, cache_mode)
        except Exception as exc:
            rec["status"] = f"skip:dry-run-failed ({exc})"
            results.append(rec); continue
        rec["key"] = key
        entry = stage_cache.cache_entry_dir(cache_dir, key)
        if entry.exists():
            rec["status"] = "already-present"
            results.append(rec); continue
        if not apply:
            rec["status"] = "would-revive"
            results.append(rec); continue
        write_entry(run_dir, cache_dir, key, cache_mode)
        # verify: a fresh dry-run must now report a hit under the new key
        _, status2 = dry_run_key(spec["module"], argv, cache_dir, cache_mode)
        rec["status"] = "revived+verified" if status2 == "hit" else f"revived-but-verify={status2}"
        results.append(rec)
    return results


_TMP = tempfile.TemporaryDirectory(prefix="rekey-out-")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_roots", nargs="+", type=Path, help="Project/run root dirs (each has openai-runs/).")
    ap.add_argument("--cache-mode", default="query-only", choices=("query-only", "strict"),
                    help="Key mode to revive under. query-only (default) also survives future refactors; "
                         "future pipeline runs must pass the same --cache-mode to hit it.")
    ap.add_argument("--apply", action="store_true", help="Write entries (default: report only).")
    args = ap.parse_args(argv)
    total = Counter()
    for root in args.run_roots:
        root = root.expanduser().resolve()
        print(f"\n== {root} ==")
        for rec in process_run(root, args.cache_mode, args.apply):
            total[rec["status"].split(" ")[0]] += 1
            line = f"  [{rec['stage']:>8}] {rec['status']}"
            if rec.get("key"):
                line += f"  key={rec['key'][:12]}"
            print(line)
            for off in rec.get("offending", []):
                print(f"             diff: {off[:120]}")
    print("\n== summary ==")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
