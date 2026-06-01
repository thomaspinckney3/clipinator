from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from extractor.transcribe import load_env_file


DEFAULT_ENV_FILE = ".env"
DEFAULT_API_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_RECORDINGS_API_BASE_URL = "https://dash.cloudflare.com/api/v4"
DEFAULT_OUTPUT_DIR = Path("cloudflare-captures")
DEFAULT_INDEX_FILENAME = "cloudflare-stream-index.json"
DEFAULT_POLL_INTERVAL_SECONDS = 10.0
DEFAULT_TIMEOUT_SECONDS = 30 * 60.0
DOWNLOAD_TYPE_CHOICES = ("mp4", "audio")
CAPTURE_METHOD_CHOICES = ("download", "playback")


@dataclass(frozen=True)
class CloudflareConfig:
    account_id: str | None
    api_token: str
    api_base_url: str
    recordings_api_base_url: str


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def safe_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "-" for char in value.strip()
    )
    return cleaned.strip(".-") or "cloudflare-stream"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_config(args: argparse.Namespace) -> CloudflareConfig:
    account_id = args.account_id or env_value("CLOUDFLARE_ACCOUNT_ID")
    api_token = args.api_token or env_value("CLOUDFLARE_API_TOKEN")
    if not api_token:
        raise RuntimeError("Cloudflare API token is required via --api-token or CLOUDFLARE_API_TOKEN.")
    return CloudflareConfig(
        account_id=account_id,
        api_token=api_token,
        api_base_url=(args.api_base_url or DEFAULT_API_BASE_URL).rstrip("/"),
        recordings_api_base_url=(
            args.recordings_api_base_url or DEFAULT_RECORDINGS_API_BASE_URL
        ).rstrip("/"),
    )


def resolve_video_uid_arg(args: argparse.Namespace) -> str | None:
    if args.video_uid and args.stream_id and args.video_uid != args.stream_id:
        raise RuntimeError("--video-uid and --stream-id refer to the same value; pass only one.")
    return args.video_uid or args.stream_id


class CloudflareStreamClient:
    def __init__(self, config: CloudflareConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {config.api_token}"},
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> Any:
        url = f"{(base_url or self.config.api_base_url).rstrip('/')}/{path.lstrip('/')}"
        request_kwargs: dict[str, Any] = {}
        if json_body is not None:
            request_kwargs["json"] = json_body
        response = self.client.request(method, url, **request_kwargs)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Cloudflare API returned non-JSON response {response.status_code} for {method} {url}."
            ) from exc

        if response.status_code >= 400 or not payload.get("success", False):
            errors = payload.get("errors") or []
            messages = payload.get("messages") or []
            details = "; ".join(
                str(item.get("message") if isinstance(item, dict) else item)
                for item in [*errors, *messages]
                if item
            )
            raise RuntimeError(
                f"Cloudflare API request failed ({response.status_code}) for {method} {url}: "
                f"{details or payload}"
            )
        return payload.get("result")

    def list_accounts(self) -> list[dict[str, Any]]:
        result = self.request("GET", "/accounts")
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected accounts response: {result!r}")
        return [item for item in result if isinstance(item, dict)]

    def account_id(self) -> str:
        if not self.config.account_id:
            raise RuntimeError("Cloudflare account id is required.")
        return self.config.account_id

    def video_details(self, video_uid: str) -> dict[str, Any]:
        result = self.request(
            "GET",
            f"/accounts/{self.account_id()}/stream/{video_uid}",
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected video response for {video_uid}: {result!r}")
        return result

    def live_input_recordings(self, live_input_uid: str) -> list[dict[str, Any]]:
        result = self.request(
            "GET",
            f"/accounts/{self.account_id()}/stream/live_inputs/{live_input_uid}/videos",
            base_url=self.config.recordings_api_base_url,
        )
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected recordings response for {live_input_uid}: {result!r}")
        return [item for item in result if isinstance(item, dict)]

    def downloads(self, video_uid: str) -> dict[str, Any]:
        result = self.request(
            "GET",
            f"/accounts/{self.account_id()}/stream/{video_uid}/downloads",
        )
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected downloads response for {video_uid}: {result!r}")
        return result

    def create_download(self, video_uid: str, download_type: str) -> dict[str, Any]:
        suffix = "audio" if download_type == "audio" else ""
        path = f"/accounts/{self.account_id()}/stream/{video_uid}/downloads"
        if suffix:
            path = f"{path}/{suffix}"
        result = self.request("POST", path)
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected download creation response for {video_uid}: {result!r}")
        return result

    def playback_token(self, video_uid: str) -> str:
        result = self.request(
            "POST",
            f"/accounts/{self.account_id()}/stream/{video_uid}/token",
        )
        if not isinstance(result, dict) or not result.get("token"):
            raise RuntimeError(f"Unexpected playback token response for {video_uid}: {result!r}")
        return str(result["token"])


def status_key(download_type: str) -> str:
    return "audio" if download_type == "audio" else "default"


def video_state(video: dict[str, Any]) -> str:
    status = video.get("status") if isinstance(video.get("status"), dict) else {}
    return str(status.get("state") or "")


def recording_sort_key(recording: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(recording.get("created") or ""),
        str(recording.get("uploaded") or ""),
        str(recording.get("uid") or ""),
    )


def select_recording(
    recordings: list[dict[str, Any]],
    *,
    latest: bool,
    video_uid: str | None,
    require_ready: bool,
) -> dict[str, Any]:
    candidates = recordings
    if video_uid:
        candidates = [item for item in candidates if item.get("uid") == video_uid]
    if require_ready:
        candidates = [item for item in candidates if video_state(item) == "ready"]
    if not candidates:
        suffix = " matching video uid" if video_uid else ""
        readiness = " ready" if require_ready else ""
        raise RuntimeError(f"No{readiness}{suffix} live-input recordings found.")
    candidates = sorted(candidates, key=recording_sort_key, reverse=True)
    if latest or len(candidates) == 1:
        return candidates[0]
    lines = "\n".join(f"  {item.get('uid')}  {item.get('created')}  {video_state(item)}" for item in candidates)
    raise RuntimeError(
        "Multiple recordings matched. Pass --latest-ready or --video-uid.\n"
        f"Candidates:\n{lines}"
    )


def print_recordings(recordings: list[dict[str, Any]]) -> None:
    for item in sorted(recordings, key=recording_sort_key, reverse=True):
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        print(
            "\t".join(
                [
                    str(item.get("uid") or ""),
                    video_state(item) or "unknown",
                    str(item.get("duration") or ""),
                    str(item.get("created") or ""),
                    str(meta.get("name") or ""),
                ]
            )
        )


def print_accounts(accounts: list[dict[str, Any]]) -> None:
    for item in accounts:
        print(
            "\t".join(
                [
                    str(item.get("id") or ""),
                    str(item.get("name") or ""),
                    str(item.get("type") or ""),
                ]
            )
        )


def account_id_from_accounts(accounts: list[dict[str, Any]]) -> str:
    if not accounts:
        raise RuntimeError(
            "The API token could not list any Cloudflare accounts. "
            "Pass --account-id explicitly or use a token with account access."
        )
    if len(accounts) == 1:
        account_id = accounts[0].get("id")
        if not account_id:
            raise RuntimeError(f"Cloudflare returned an account without an id: {accounts[0]!r}")
        return str(account_id)
    lines = "\n".join(
        f"  {item.get('id')}  {item.get('name') or ''}" for item in accounts
    )
    raise RuntimeError(
        "The API token can access multiple Cloudflare accounts. "
        "Pass --account-id or run --list-accounts and choose one.\n"
        f"Accounts:\n{lines}"
    )


def config_with_account_id(config: CloudflareConfig, account_id: str) -> CloudflareConfig:
    return CloudflareConfig(
        account_id=account_id,
        api_token=config.api_token,
        api_base_url=config.api_base_url,
        recordings_api_base_url=config.recordings_api_base_url,
    )


def desired_download(downloads: dict[str, Any], download_type: str) -> dict[str, Any] | None:
    value = downloads.get(status_key(download_type))
    return value if isinstance(value, dict) else None


def wait_for_download(
    client: CloudflareStreamClient,
    video_uid: str,
    download_type: str,
    *,
    generate: bool,
    wait: bool,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    downloads = client.downloads(video_uid)
    item = desired_download(downloads, download_type)
    if item is None and generate:
        downloads = client.create_download(video_uid, download_type)
        item = desired_download(downloads, download_type)
    if item is None:
        raise RuntimeError(
            f"No {download_type} download exists for {video_uid}. "
            "Pass --generate-download to create one."
        )

    started = time.monotonic()
    while True:
        state = str(item.get("status") or "")
        percent = item.get("percentComplete")
        if state == "ready":
            url = item.get("url")
            if not url:
                raise RuntimeError(f"Cloudflare marked the download ready but did not return a URL.")
            return item
        if state == "error":
            raise RuntimeError(f"Cloudflare download generation failed for {video_uid}: {item}")
        if not wait:
            raise RuntimeError(
                f"Download is not ready yet: status={state or 'unknown'} "
                f"percent={percent if percent is not None else 'unknown'}."
            )
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for {download_type} download for {video_uid}: "
                f"status={state or 'unknown'} percent={percent if percent is not None else 'unknown'}."
            )
        print(
            f"Waiting for {download_type} download: status={state or 'unknown'} "
            f"percent={percent if percent is not None else 'unknown'}",
            file=sys.stderr,
        )
        time.sleep(poll_interval_seconds)
        downloads = client.downloads(video_uid)
        item = desired_download(downloads, download_type)
        if item is None:
            raise RuntimeError(f"Download disappeared while polling for {video_uid}.")


def output_path_for(
    output_dir: Path,
    video_uid: str,
    download_type: str,
    *,
    video: dict[str, Any] | None,
    recording: dict[str, Any] | None,
) -> Path:
    meta = {}
    source = recording or video or {}
    if isinstance(source.get("meta"), dict):
        meta = source["meta"]
    raw_name = str(meta.get("name") or source.get("uid") or video_uid)
    suffix = ".m4a" if download_type == "audio" else ".mp4"
    return output_dir / f"{safe_name(raw_name)}-{video_uid}{suffix}"


def playback_hls_url(video: dict[str, Any], video_uid: str, token: str | None = None) -> str:
    playback = video.get("playback") if isinstance(video.get("playback"), dict) else {}
    hls_url = playback.get("hls")
    if not hls_url:
        hls_url = f"https://videodelivery.net/{video_uid}/manifest/video.m3u8"
    hls_url = str(hls_url)
    if token:
        return hls_url.replace(f"/{video_uid}/manifest/", f"/{token}/manifest/")
    return hls_url


def ffmpeg_headers(referer: str | None, origin: str | None) -> str | None:
    lines: list[str] = []
    if referer:
        lines.append(f"Referer: {referer}")
    if origin:
        lines.append(f"Origin: {origin}")
    if not lines:
        return None
    return "\r\n".join(lines) + "\r\n"


def capture_playback_stream(
    manifest_url: str,
    output_path: Path,
    *,
    download_type: str,
    force: bool,
    ffmpeg_bin: str,
    referer: str | None,
    origin: str | None,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --force to replace it.")
    resolved_ffmpeg = shutil.which(ffmpeg_bin)
    if resolved_ffmpeg is None:
        raise RuntimeError(f"Required command not found on PATH: {ffmpeg_bin}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f".partial{output_path.suffix}")
    command = [
        resolved_ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    headers = ffmpeg_headers(referer, origin)
    if headers:
        command.extend(["-headers", headers])
    command.extend(["-i", manifest_url])
    if download_type == "audio":
        command.extend(["-map", "0:a:0?", "-vn", "-sn", "-c:a", "copy"])
    else:
        command.extend(["-map", "0:v:0?", "-map", "0:a:0?", "-sn", "-c", "copy", "-movflags", "+faststart"])
    command.append(str(temp_path))

    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        message = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(message)
    temp_path.replace(output_path)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "ffmpeg_bin": resolved_ffmpeg,
        "capture_method": "playback",
    }


def download_file(url: str, output_path: Path, *, force: bool) -> dict[str, Any]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --force to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".partial")
    downloaded = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        content_length = response.headers.get("content-length")
        with temp_path.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
    temp_path.replace(output_path)
    return {
        "path": str(output_path),
        "bytes": downloaded,
        "content_length": int(content_length) if content_length and content_length.isdigit() else None,
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def output_dir_for_args(args: argparse.Namespace) -> Path:
    return args.output_dir or Path(env_value("CLOUDFLARE_CAPTURE_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)


def index_path_for_args(args: argparse.Namespace, output_dir: Path | None = None) -> Path:
    if args.index_output:
        return args.index_output.expanduser().resolve()
    env_path = env_value("CLOUDFLARE_CAPTURE_INDEX")
    if env_path:
        return Path(env_path).expanduser().resolve()
    root = output_dir or output_dir_for_args(args)
    return root.expanduser().resolve() / DEFAULT_INDEX_FILENAME


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_capture_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "streams": {},
        }
    data = read_json(path)
    if not isinstance(data.get("streams"), dict):
        data["streams"] = {}
    return data


def capture_record_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    video = manifest.get("video") if isinstance(manifest.get("video"), dict) else {}
    meta = video.get("meta") if isinstance(video.get("meta"), dict) else {}
    file_info = manifest.get("file") if isinstance(manifest.get("file"), dict) else {}
    return {
        "captured_at": manifest.get("created_at"),
        "stream_id": manifest.get("video_uid"),
        "account_id": manifest.get("account_id"),
        "live_input_uid": manifest.get("live_input_uid"),
        "capture_method": manifest.get("capture_method"),
        "download_type": manifest.get("download_type"),
        "file": file_info.get("path"),
        "metadata": manifest.get("metadata_path"),
        "bytes": file_info.get("bytes"),
        "cloudflare_name": meta.get("name"),
        "cloudflare_duration_seconds": video.get("duration"),
        "ready_to_stream": video.get("readyToStream"),
        "require_signed_urls": video.get("requireSignedURLs"),
    }


def update_capture_index(index_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    video_uid = str(manifest.get("video_uid") or "").strip()
    if not video_uid:
        raise RuntimeError("Cannot index capture metadata without video_uid.")
    record = capture_record_from_manifest(manifest)
    index = load_capture_index(index_path)
    streams = index.setdefault("streams", {})
    stream = streams.get(video_uid)
    if not isinstance(stream, dict):
        stream = {
            "stream_id": video_uid,
            "downloads": [],
        }
        streams[video_uid] = stream

    stream.update(
        {
            "stream_id": video_uid,
            "account_id": record.get("account_id"),
            "cloudflare_name": record.get("cloudflare_name"),
            "cloudflare_duration_seconds": record.get("cloudflare_duration_seconds"),
            "ready_to_stream": record.get("ready_to_stream"),
            "require_signed_urls": record.get("require_signed_urls"),
            "latest_file": record.get("file"),
            "latest_metadata": record.get("metadata"),
            "latest_capture_method": record.get("capture_method"),
            "latest_download_type": record.get("download_type"),
            "latest_captured_at": record.get("captured_at"),
        }
    )

    downloads = stream.setdefault("downloads", [])
    if not isinstance(downloads, list):
        downloads = []
        stream["downloads"] = downloads
    record_key = (record.get("file"), record.get("metadata"), record.get("capture_method"))
    kept = [
        item
        for item in downloads
        if not (
            isinstance(item, dict)
            and (item.get("file"), item.get("metadata"), item.get("capture_method")) == record_key
        )
    ]
    kept.append(record)
    stream["downloads"] = kept
    index["updated_at"] = utc_now()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index


def record_metadata_files(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = output_dir_for_args(args)
    index_path = index_path_for_args(args, output_dir)
    recorded: list[str] = []
    for raw_path in args.record_metadata:
        metadata_path = raw_path.expanduser().resolve()
        manifest = read_json(metadata_path)
        manifest.setdefault("metadata_path", str(metadata_path))
        update_capture_index(index_path, manifest)
        recorded.append(str(metadata_path))
    return {
        "action": "record-metadata",
        "index_path": str(index_path),
        "metadata_files": recorded,
        "record_count": len(recorded),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(args.env_file)
    if args.record_metadata:
        return record_metadata_files(args)

    config = resolve_config(args)
    account_probe = CloudflareStreamClient(config)
    try:
        if args.list_accounts:
            accounts = account_probe.list_accounts()
            print_accounts(accounts)
            return {
                "action": "list-accounts",
                "account_count": len(accounts),
            }
        if not config.account_id:
            config = config_with_account_id(config, account_id_from_accounts(account_probe.list_accounts()))
    finally:
        account_probe.close()

    requested_video_uid = resolve_video_uid_arg(args)
    if not requested_video_uid and not args.live_input_uid:
        raise RuntimeError("Pass --stream-id, --video-uid, or --live-input-uid.")
    account_id = str(config.account_id)
    if args.dry_run and requested_video_uid and not args.live_input_uid:
        download_path = f"/accounts/{account_id}/stream/{requested_video_uid}/downloads"
        if args.type == "audio":
            download_path = f"{download_path}/audio"
        requests: list[dict[str, str]] = [
            {
                "method": "GET",
                "path": f"/accounts/{account_id}/stream/{requested_video_uid}",
            },
        ]
        if args.capture_method == "download":
            requests.append(
                {
                    "method": "GET",
                    "path": f"/accounts/{account_id}/stream/{requested_video_uid}/downloads",
                }
            )
            if args.generate_download:
                requests.append({"method": "POST", "path": download_path})
        return {
            "action": "dry-run",
            "account_id": account_id,
            "video_uid": requested_video_uid,
            "capture_method": args.capture_method,
            "download_type": args.type,
            "api_base_url": config.api_base_url,
            "planned_requests": requests,
            "planned_local_tool": "ffmpeg from playback.hls" if args.capture_method == "playback" else None,
        }

    client = CloudflareStreamClient(config)
    try:
        recording: dict[str, Any] | None = None
        video_uid = requested_video_uid
        if args.live_input_uid:
            recordings = client.live_input_recordings(args.live_input_uid)
            if args.list_recordings:
                print_recordings(recordings)
                return {
                    "action": "list-recordings",
                    "live_input_uid": args.live_input_uid,
                    "recording_count": len(recordings),
                }
            recording = select_recording(
                recordings,
                latest=args.latest_ready,
                video_uid=requested_video_uid,
                require_ready=True,
            )
            video_uid = str(recording["uid"])

        if not video_uid:
            raise RuntimeError("Unable to determine Cloudflare Stream video uid.")

        video = client.video_details(video_uid)
        if video_state(video) not in {"ready", ""} and not bool(video.get("readyToStream")):
            raise RuntimeError(f"Video {video_uid} is not ready for download: {video.get('status')}")

        if args.dry_run:
            return {
                "action": "dry-run",
                "account_id": account_id,
                "video_uid": video_uid,
                "capture_method": args.capture_method,
                "download_type": args.type,
                "video": video,
                "recording": recording,
            }

        output_dir = output_dir_for_args(args)
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else output_path_for(
                output_dir.expanduser().resolve(),
                video_uid,
                args.type,
                video=video,
                recording=recording,
            )
        )
        if args.capture_method == "playback":
            playback_token: str | None = None
            if bool(video.get("requireSignedURLs")):
                if not args.generate_playback_token:
                    raise RuntimeError(
                        "This video requires signed playback URLs. "
                        "Pass --generate-playback-token or provide a public/playable stream."
                    )
                playback_token = client.playback_token(video_uid)
            manifest_url = playback_hls_url(video, video_uid, playback_token)
            file_info = capture_playback_stream(
                manifest_url,
                output_path,
                download_type=args.type,
                force=args.force,
                ffmpeg_bin=args.ffmpeg_bin,
                referer=args.referer,
                origin=args.origin,
            )
            manifest = {
                "schema_version": 1,
                "created_at": utc_now(),
                "account_id": account_id,
                "video_uid": video_uid,
                "live_input_uid": args.live_input_uid,
                "capture_method": "playback",
                "download_type": args.type,
                "playback": {
                    "hls": "<signed-token-redacted>" if playback_token else manifest_url,
                    "used_generated_token": playback_token is not None,
                },
                "file": file_info,
                "video": video,
                "recording": recording,
            }
            if args.write_metadata:
                metadata_path = args.metadata_output or output_path.with_suffix(output_path.suffix + ".json")
                manifest["metadata_path"] = str(metadata_path.expanduser().resolve())
                write_metadata(metadata_path.expanduser().resolve(), manifest)
            if args.update_index:
                index_path = index_path_for_args(args, output_dir)
                update_capture_index(index_path, manifest)
                manifest["index_path"] = str(index_path)
            return manifest

        download = wait_for_download(
            client,
            video_uid,
            args.type,
            generate=args.generate_download,
            wait=args.wait,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        file_info = download_file(str(download["url"]), output_path, force=args.force)
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "account_id": account_id,
            "video_uid": video_uid,
            "live_input_uid": args.live_input_uid,
            "capture_method": "download",
            "download_type": args.type,
            "download": download,
            "file": file_info,
            "video": video,
            "recording": recording,
        }
        if args.write_metadata:
            metadata_path = args.metadata_output or output_path.with_suffix(output_path.suffix + ".json")
            manifest["metadata_path"] = str(metadata_path.expanduser().resolve())
            write_metadata(metadata_path.expanduser().resolve(), manifest)
        if args.update_index:
            index_path = index_path_for_args(args, output_dir)
            update_capture_index(index_path, manifest)
            manifest["index_path"] = str(index_path)
        return manifest
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture Cloudflare Stream videos or live-input recordings via the Stream API."
    )
    parser.add_argument("--env-file", type=Path, default=Path(DEFAULT_ENV_FILE))
    parser.add_argument("--account-id", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--recordings-api-base-url", default=DEFAULT_RECORDINGS_API_BASE_URL)
    parser.add_argument(
        "--list-accounts",
        action="store_true",
        help="List Cloudflare accounts visible to the API token and exit.",
    )
    parser.add_argument(
        "--stream-id",
        default=None,
        help="Cloudflare Stream video UID to download. Alias for --video-uid.",
    )
    parser.add_argument("--video-uid", default=None, help="Cloudflare Stream video UID to download.")
    parser.add_argument(
        "--live-input-uid",
        default=None,
        help="Cloudflare Stream live input UID. The utility selects a ready recording from it.",
    )
    parser.add_argument(
        "--latest-ready",
        action="store_true",
        help="When using --live-input-uid, select the newest ready recording.",
    )
    parser.add_argument(
        "--list-recordings",
        action="store_true",
        help="List recordings for --live-input-uid instead of downloading.",
    )
    parser.add_argument(
        "--capture-method",
        choices=CAPTURE_METHOD_CHOICES,
        default="download",
        help=(
            "download uses Cloudflare's generated MP4/M4A download asset; "
            "playback remuxes the HLS playback manifest with ffmpeg."
        ),
    )
    parser.add_argument("--type", choices=DOWNLOAD_TYPE_CHOICES, default="mp4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--index-output",
        type=Path,
        default=None,
        help=(
            "Path to the Stream ID capture index. Defaults to "
            "CLOUDFLARE_CAPTURE_INDEX or cloudflare-stream-index.json in the output dir."
        ),
    )
    parser.add_argument(
        "--update-index",
        dest="update_index",
        action="store_true",
        default=True,
        help="Update the local Stream ID to download index after a successful capture.",
    )
    parser.add_argument(
        "--no-update-index",
        dest="update_index",
        action="store_false",
        help="Do not update the local Stream ID capture index.",
    )
    parser.add_argument(
        "--record-metadata",
        type=Path,
        action="append",
        default=[],
        help="Add an existing capture metadata JSON file to the local index and exit.",
    )
    parser.add_argument(
        "--generate-download",
        dest="generate_download",
        action="store_true",
        default=True,
        help="Create the Cloudflare downloadable asset when it does not already exist.",
    )
    parser.add_argument(
        "--no-generate-download",
        dest="generate_download",
        action="store_false",
        help="Only use an existing Cloudflare downloadable asset.",
    )
    parser.add_argument("--wait", dest="wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    parser.add_argument("--poll-interval-seconds", type=float, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--generate-playback-token",
        action="store_true",
        help="For private signed Stream videos, generate a temporary playback token for HLS capture.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--referer", default=None, help="Optional Referer header for playback capture.")
    parser.add_argument("--origin", default=None, help="Optional Origin header for playback capture.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-metadata", action="store_true", default=True)
    parser.add_argument("--no-write-metadata", dest="write_metadata", action="store_false")
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = capture(args)
        if args.list_accounts or args.list_recordings:
            return 0
        if manifest.get("action") == "record-metadata":
            print(f"Updated index: {manifest['index_path']}")
            print(f"Recorded metadata files: {manifest['record_count']}")
            return 0
        if args.dry_run:
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return 0
        file_info = manifest["file"]
        print(f"Wrote {file_info['path']}")
        print(f"Bytes: {file_info['bytes']}")
        if manifest.get("metadata_path"):
            print(f"Metadata: {manifest['metadata_path']}")
        if manifest.get("index_path"):
            print(f"Index: {manifest['index_path']}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
