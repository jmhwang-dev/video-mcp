#!/usr/bin/env python3
"""Download a YouTube video (video-only) at an interactively chosen resolution.

The file is saved into the current directory and is intended for scene-transition
(shot boundary) detection, so only the video stream is fetched (no audio).

Usage:
    uv run download_video.py <youtube-url>            # interactive resolution menu
    uv run download_video.py <youtube-url> --height 720   # skip the menu
    uv run download_video.py <youtube-url> --list     # just list resolutions and exit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yt_dlp import YoutubeDL

OUTPUT_TEMPLATE = "%(title)s [%(id)s] [%(height)sp].%(ext)s"


def human_size(num_bytes: int | None) -> str:
    if not num_bytes:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def short_codec(vcodec: str | None) -> str:
    """Normalise a yt-dlp vcodec string to a short, NVDEC-friendly label."""
    if not vcodec or vcodec == "none":
        return "?"
    v = vcodec.lower()
    if v.startswith("avc") or "h264" in v:
        return "h264"
    if v.startswith(("hev", "hvc")) or "h265" in v:
        return "hevc"
    if v.startswith(("vp9", "vp09")):
        return "vp9"
    if v.startswith("av01"):
        return "av1"
    return v.split(".")[0]


def codec_pref(vcodec: str | None) -> int:
    """Lower is better. Prefer NVDEC-friendly codecs (h264 > hevc > vp9 > av1)."""
    return {"h264": 0, "hevc": 1, "vp9": 2, "av1": 3}.get(short_codec(vcodec), 4)


def best_format_per_height(info: dict) -> dict[int, dict]:
    """Pick one representative video-only format per resolution height."""
    video_only = [
        f
        for f in info.get("formats", [])
        if f.get("vcodec") not in (None, "none")
        and f.get("acodec") in (None, "none")
        and f.get("height")
    ]
    # Best per height: prefer NVDEC-friendly codec, then higher bitrate.
    video_only.sort(key=lambda f: (codec_pref(f.get("vcodec")), -(f.get("tbr") or 0)))
    reps: dict[int, dict] = {}
    for f in video_only:
        reps.setdefault(int(f["height"]), f)
    return reps


def print_resolution_table(reps: dict[int, dict]) -> list[dict]:
    rows = sorted(reps.values(), key=lambda f: f["height"], reverse=True)
    print("\n사용 가능한 해상도 (video-only):\n")
    print(f"  {'#':>2}  {'해상도':>8}  {'fps':>4}  {'코덱':<6} {'확장자':<5} {'대략 용량':>10}")
    print("  " + "-" * 52)
    for i, f in enumerate(rows, 1):
        fps = f.get("fps")
        fps_s = f"{fps:g}" if fps else ""
        size = human_size(f.get("filesize") or f.get("filesize_approx"))
        print(
            f"  {i:>2}  {str(f['height']) + 'p':>8}  {fps_s:>4}  "
            f"{short_codec(f.get('vcodec')):<6} {f.get('ext', ''):<5} {size:>10}"
        )
    print()
    return rows


def choose_height_interactive(reps: dict[int, dict]) -> int:
    rows = print_resolution_table(reps)
    while True:
        try:
            raw = input("다운로드할 번호를 선택하세요 (q=취소): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n취소되었습니다.")
            sys.exit(1)
        if raw.lower() in ("q", "quit", "exit"):
            print("취소되었습니다.")
            sys.exit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            return int(rows[int(raw) - 1]["height"])
        print(f"  1 ~ {len(rows)} 사이의 번호 또는 q 를 입력하세요.")


def extract_info(url: str) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    with YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def download(url: str, height: int) -> str | None:
    # Prefer an exact-height mp4 (NVDEC-friendly), then any codec at that height,
    # then the best video-only stream at or below that height.
    fmt = (
        f"bv*[height={height}][ext=mp4]/"
        f"bv*[height={height}]/"
        f"bv*[height<={height}]"
    )
    opts = {
        "format": fmt,
        "outtmpl": OUTPUT_TEMPLATE,
        "noplaylist": True,
        "paths": {"home": str(Path.cwd())},
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloads = info.get("requested_downloads") or []
        if downloads:
            return downloads[0].get("filepath")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video (video-only) at a chosen resolution."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Target resolution height (e.g. 1080). Skips the interactive menu.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available resolutions and exit (no download).",
    )
    args = parser.parse_args()

    print(f"메타데이터를 가져오는 중: {args.url}")
    info = extract_info(args.url)
    title = info.get("title", "?")
    duration = info.get("duration")
    dur_s = f"{int(duration // 60)}분 {int(duration % 60)}초" if duration else "?"
    print(f"제목: {title}  |  길이: {dur_s}")

    reps = best_format_per_height(info)
    if not reps:
        print("video-only 포맷을 찾지 못했습니다. URL 또는 영상 사용 가능 여부를 확인하세요.")
        sys.exit(1)

    if args.list:
        print_resolution_table(reps)
        return

    if args.height is not None:
        available = sorted(reps, reverse=True)
        if args.height in reps:
            height = args.height
        else:
            # Snap to the nearest available height at or below the request.
            at_or_below = [h for h in available if h <= args.height]
            height = at_or_below[0] if at_or_below else available[-1]
            print(f"요청한 {args.height}p 가 없어 {height}p 로 진행합니다.")
    else:
        height = choose_height_interactive(reps)

    print(f"\n{height}p (video-only) 다운로드를 시작합니다...\n")
    path = download(args.url, height)
    if path:
        print(f"\n저장 완료: {path}")
    else:
        print("\n다운로드는 끝났지만 파일 경로를 확인하지 못했습니다. 현재 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
