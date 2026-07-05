import json
from pathlib import Path

from extractor import project_layout as pl

from conftest import make_clip, make_plan


class TestSlugs:
    def test_source_slug_strips_hash_and_variant(self):
        assert pl.source_slug(Path("Ergo-Gemes-1.3-260605_master.mp4-132286bc678f9609d858b4791ef4436c.mp4")) == "ergo-gemes-1-3-260605"
        assert pl.source_slug(Path("yale-psyc110-lecture-1-introduction-excerpt-insight-20260507T204911Z.mp4")) == "yale-psyc110-lecture-1-introduction"

    def test_clip_slug_orders_and_humanizes(self):
        assert pl.clip_slug("Easy to check. Hard to solve?", 1) == "01-easy-to-check-hard-to-solve"
        assert pl.clip_slug("", 3).startswith("03-")

    def test_clip_index_from_plan(self):
        plan = make_plan([make_clip("EX01"), make_clip("EX02")])
        idx = pl.clip_index(plan)
        assert idx["EX01"][0] == 1 and idx["EX02"][0] == 2
        assert idx["EX01"][2] == "01-title-ex01"


class TestEstablishMedia:
    def test_links_master_and_sidecar(self, tmp_path):
        src = tmp_path / "orig.mp4"; src.write_bytes(b"video-bytes")
        (Path(str(src) + ".json")).write_text('{"capture": true}')
        proj = tmp_path / "proj"
        media = pl.establish_media(proj, src)
        assert media == proj / "media" / "source.mp4"
        assert media.read_bytes() == b"video-bytes"
        assert (proj / "media" / "source.capture.json").exists()

    def test_idempotent(self, tmp_path):
        src = tmp_path / "orig.mp4"; src.write_bytes(b"x")
        proj = tmp_path / "proj"
        a = pl.establish_media(proj, src)
        b = pl.establish_media(proj, src)  # no error second time
        assert a == b


def _fake_render(root: Path, clip_ids: list[str]) -> Path:
    """A render dir with a manifest and per-clip output files (no ffmpeg)."""
    rdir = root / "_renders" / "final"; rdir.mkdir(parents=True)
    clips = []
    for cid in clip_ids:
        for suf in (".mp4", ".subtitled.mp4", ".srt"):
            (rdir / f"{cid}{suf}").write_text(f"{cid}{suf}")
        clips.append({"clip_id": cid, "output_path": str(rdir / f"{cid}.mp4")})
    (rdir / "render-manifest.json").write_text(json.dumps({"clips": clips}))
    return rdir / "render-manifest.json"


class TestDeliverablesDirname:
    def test_mode_named_folders(self):
        assert pl.deliverables_dirname("excerpt") == "excerpts"
        assert pl.deliverables_dirname("ad") == "ads"


class TestBuildClips:
    def test_creates_human_named_clip_dirs_under_mode_folder(self, tmp_path):
        proj = tmp_path / "proj"; proj.mkdir()
        plan = make_plan([make_clip("EX01"), make_clip("EX02")])
        rm = _fake_render(proj, ["EX01", "EX02"])
        index = pl.build_clips(proj, rm, plan, "excerpt")
        d = proj / "excerpts" / "01-title-ex01"
        assert (d / "final.mp4").exists()
        assert (d / "final.subtitled.mp4").exists()
        assert (d / "captions.srt").exists()
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["clip_id"] == "EX01" and meta["order"] == 1
        assert [e["slug"] for e in index] == ["01-title-ex01", "02-title-ex02"]

    def test_ad_mode_uses_ads_folder(self, tmp_path):
        proj = tmp_path / "proj"; proj.mkdir()
        plan = make_plan([make_clip("EX01")])
        rm = _fake_render(proj, ["EX01"])
        pl.build_clips(proj, rm, plan, "ad")
        assert (proj / "ads" / "01-title-ex01" / "final.mp4").exists()
        assert not (proj / "extracts").exists()

    def test_idempotent(self, tmp_path):
        proj = tmp_path / "proj"; proj.mkdir()
        plan = make_plan([make_clip("EX01")])
        rm = _fake_render(proj, ["EX01"])
        pl.build_clips(proj, rm, plan, "excerpt")
        pl.build_clips(proj, rm, plan, "excerpt")  # second run must not raise
        assert (proj / "excerpts" / "01-title-ex01" / "final.mp4").exists()


class TestGenerateLayout:
    def test_writes_project_json_with_clip_index(self, tmp_path):
        proj = tmp_path / "ergo-demo"; proj.mkdir()
        plan = make_plan([make_clip("EX01")])
        rm = _fake_render(proj, ["EX01"])
        pl.generate_layout(proj, rm, plan, "ergo-demo", content_mode="excerpt",
                           original_master="orig.mp4", promoted_from="ergo-demo",
                           created_at="2026-07-04T00:00:00+00:00")
        pj = json.loads((proj / "project.json").read_text())
        assert pj["source"]["slug"] == "ergo-demo"
        assert pj["source"]["master"] == "media/source.mp4"
        assert pj["source"]["original_master"] == "orig.mp4"
        assert pj["content_mode"] == "excerpt"
        assert pj["clips_dir"] == "excerpts"
        assert len(pj["clips"]) == 1

    def test_no_render_manifest_still_writes_project_json(self, tmp_path):
        proj = tmp_path / "p"; proj.mkdir()
        pl.generate_layout(proj, None, make_plan(), "p", content_mode="ad",
                           original_master="o.mp4", promoted_from="p")
        assert (proj / "project.json").exists()
        assert json.loads((proj / "project.json").read_text())["clips"] == []
