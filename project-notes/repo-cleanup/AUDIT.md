# Pipeline and Web Interface Cleanup Audit

## 2026-08-19

### Verdict

- The cleanup passes functional, structural, and forward-impact review for the agreed pipeline-and-web scope. Experimental probe code, the superseded deterministic planner, generated capture metadata, unused Homebrew packages, and stale deployment scaffolding were removed. Pipeline preprocessing retains its standard transcript-QA behavior, and the OTIO implementation, dependencies, CLI, documentation, and tests remain intact. The change is exposed in the source tree and packaging metadata; it has not been release-verified by installing or running a production pipeline job.

### Findings

- No cleanup defects found. OTIO export remains CLI-only in the current implementation; this change preserves it but does not add a browser route or button.
  - evidence:
    - `pyproject.toml:36`
    - `README.md:936`
    - `src/extractor/review_server.py:765`
  - commands used:
    - `rg -n 'otio_edl|otio-edl|opentimelineio' src/extractor/review_server.py src/extractor/review_static/index.html README.md pyproject.toml requirements.txt tests`

### Completeness

- Truthful state: exposed
- Acceptance criteria: all cleanup items addressed; OTIO explicitly preserved
- Completion matrix: pipeline, web editor, packaging, and OTIO regression surfaces proven
- Implementation approach steps: all done
- Done-when conditions: source and documentation references removed; test suites pass
- Files listed in brief: all applicable files touched; conditional migration utilities retained
- Architectural intent: achieved
- Deferred items: one-time migration utilities remain until existing projects are confirmed upgraded
- Placeholder audit: clean

### Structural Assessment

- Architecture: clean; standard QA defaults now belong to `qa_transcript.py` rather than a deleted planner prototype.
- Abstraction: clean; preprocessing consumes the QA-owned programmatic defaults.
- Semantic families: unified
- Patterns: consistent
- Integrity: clean
- Scope: clean; OTIO and web-editor components were not removed.

### Forward Impact

- Clean. Removing the prototype dependency reduces the preprocessing import surface. Keeping OTIO dependencies and tests preserves future browser integration.
- Small fix now: none

### Verification

- `git diff --check`
- `.venv/bin/python -m compileall -q src`
- `.venv/bin/pip install --no-deps -e .`
- `.venv/bin/pytest -q tests/test_qa_transcript.py tests/test_otio_edl.py tests/test_review_server.py`
- `.venv/bin/pytest -q`
