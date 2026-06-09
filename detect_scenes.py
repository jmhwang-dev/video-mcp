#!/usr/bin/env python3
"""Detect scene-transition (shot boundary) points in a video.

Two backends (see scene_segmentation.md for the methodology):

  pyscenedetect  (default, CPU): content-aware detection via PySceneDetect.
                 Robust, handles hard cuts well, works without a GPU.
  ffmpeg-gpu     (GPU decode):   ffmpeg `scene` filter with CUDA/NVDEC decoding.
                 Fast for long / high-resolution videos.

Both backends write the same artifacts to the output dir:
  scenes.csv   one row per detected scene (timecodes + frame numbers)
  cuts.json    cut points + scene list + metadata

Usage:
    uv run detect_scenes.py <video>
    uv run detect_scenes.py <video> --backend ffmpeg-gpu --threshold 0.4
    uv run detect_scenes.py <video> --detector adaptive --save-images --split
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# Per-backend default detection thresholds (different scales).
DEFAULT_THRESHOLD = {"pyscenedetect": 27.0, "ffmpeg-gpu": 0.4}


def sec_to_tc(seconds: float) -> str:
    """Seconds -> HH:MM:SS.mmm timecode."""
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3_600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def probe_video(path: str) -> tuple[float, float]:
    """Return (fps, duration_seconds) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate:format=duration",
        "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0))
    return fps, duration


# --------------------------------------------------------------------------- #
# Backend: PySceneDetect (CPU)
# --------------------------------------------------------------------------- #
def detect_pyscenedetect(
    path: str, detector: str, threshold: float, downscale: int | None, min_len_s: float
) -> list[dict]:
    from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video

    video = open_video(path)
    fps = video.frame_rate
    min_len_frames = max(1, int(round(min_len_s * fps))) if min_len_s else 15

    manager = SceneManager()
    if detector == "adaptive":
        manager.add_detector(AdaptiveDetector(min_scene_len=min_len_frames))
    else:
        manager.add_detector(
            ContentDetector(threshold=threshold, min_scene_len=min_len_frames)
        )
    if downscale:
        manager.auto_downscale = False
        manager.downscale = downscale

    manager.detect_scenes(video=video, show_progress=True)
    scene_list = manager.get_scene_list()

    # No cuts detected -> the whole video is a single scene.
    if not scene_list:
        end = video.duration or video.position
        return [
            {
                "scene": 1,
                "start_seconds": 0.0,
                "start_timecode": sec_to_tc(0.0),
                "start_frame": 0,
                "end_seconds": round(end.seconds, 3),
                "end_timecode": sec_to_tc(end.seconds),
                "end_frame": end.frame_num,
            }
        ]

    scenes: list[dict] = []
    for i, (start, end) in enumerate(scene_list, 1):
        scenes.append(
            {
                "scene": i,
                "start_seconds": round(start.seconds, 3),
                "start_timecode": sec_to_tc(start.seconds),
                "start_frame": start.frame_num,
                "end_seconds": round(end.seconds, 3),
                "end_timecode": sec_to_tc(end.seconds),
                "end_frame": end.frame_num,
            }
        )
    return scenes


# --------------------------------------------------------------------------- #
# Backend: ffmpeg scene filter with GPU (NVDEC) decoding
# --------------------------------------------------------------------------- #
def _run_ffmpeg_scene(path: str, threshold: float, use_gpu: bool) -> subprocess.CompletedProcess:
    cmd = ["ffmpeg", "-hide_banner"]
    if use_gpu:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-i", path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-an", "-f", "null", "-",
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def detect_ffmpeg_gpu(path: str, threshold: float, min_len_s: float) -> list[dict]:
    proc = _run_ffmpeg_scene(path, threshold, use_gpu=True)
    if proc.returncode != 0:
        print("  GPU 디코딩에 실패하여 CPU 디코딩으로 재시도합니다...", file=sys.stderr)
        proc = _run_ffmpeg_scene(path, threshold, use_gpu=False)
        proc.check_returncode()

    cuts = []
    for line in proc.stderr.splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = re.search(r"pts_time:([0-9.]+)", line)
            if m:
                cuts.append(float(m.group(1)))
    cuts = sorted(set(cuts))

    fps, duration = probe_video(path)
    # Build scenes from boundaries: [0, c1], [c1, c2], ..., [cn, duration].
    boundaries = [0.0]
    for c in cuts:
        if c - boundaries[-1] >= max(min_len_s, 0.0) and c < duration:
            boundaries.append(c)
    boundaries.append(duration)

    scenes: list[dict] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        scenes.append(
            {
                "scene": i + 1,
                "start_seconds": round(start, 3),
                "start_timecode": sec_to_tc(start),
                "start_frame": int(round(start * fps)),
                "end_seconds": round(end, 3),
                "end_timecode": sec_to_tc(end),
                "end_frame": int(round(end * fps)),
            }
        )
    return scenes


# --------------------------------------------------------------------------- #
# Output + optional extras
# --------------------------------------------------------------------------- #
def write_outputs(scenes: list[dict], meta: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene", "start_timecode", "start_seconds", "start_frame",
        "end_timecode", "end_seconds", "end_frame",
    ]
    with (outdir / "scenes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scenes)

    cut_points = [s["start_seconds"] for s in scenes[1:]]  # boundaries between scenes
    payload = {**meta, "num_scenes": len(scenes), "cut_points_seconds": cut_points,
               "scenes": scenes}
    (outdir / "cuts.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def save_images(scenes: list[dict], video_path: str, outdir: Path) -> None:
    """Extract one representative frame per scene (start) using ffmpeg."""
    img_dir = outdir / "scenes_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        out = img_dir / f"scene-{s['scene']:03d}.jpg"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{s['start_seconds']}", "-i", video_path,
             "-frames:v", "1", str(out)],
            check=False,
        )
    print(f"  대표 프레임 저장: {img_dir}/")


def split_clips(scenes: list[dict], video_path: str, outdir: Path, gpu_encode: bool) -> None:
    clip_dir = outdir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        out = clip_dir / f"scene-{s['scene']:03d}.mp4"
        codec = ["-c:v", "h264_nvenc"] if gpu_encode else ["-c", "copy"]
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", f"{s['start_seconds']}", "-to", f"{s['end_seconds']}",
             "-i", video_path, *codec, str(out)],
            check=False,
        )
    print(f"  구간 클립 저장: {clip_dir}/  ({'h264_nvenc' if gpu_encode else 'stream copy'})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scene-transition points in a video (CPU or GPU)."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--backend", choices=["pyscenedetect", "ffmpeg-gpu"], default="pyscenedetect",
        help="Detection backend (default: pyscenedetect, CPU).",
    )
    parser.add_argument(
        "--detector", choices=["content", "adaptive"], default="content",
        help="PySceneDetect detector (pyscenedetect backend only).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Detection threshold. Default: 27.0 (pyscenedetect) / 0.4 (ffmpeg-gpu).",
    )
    parser.add_argument(
        "--downscale", type=int, default=None,
        help="Downscale factor for speed (pyscenedetect backend only).",
    )
    parser.add_argument(
        "--min-scene-len", type=float, default=0.0,
        help="Minimum scene length in seconds (default: 0 = auto/none).",
    )
    parser.add_argument("--output-dir", default=".", help="Where to write outputs.")
    parser.add_argument("--save-images", action="store_true",
                        help="Save one representative frame per scene.")
    parser.add_argument("--split", action="store_true",
                        help="Split the video into per-scene clips.")
    parser.add_argument("--gpu-encode", action="store_true",
                        help="Use h264_nvenc when splitting (default: stream copy).")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).is_file():
        print(f"파일을 찾을 수 없습니다: {video_path}", file=sys.stderr)
        sys.exit(1)

    threshold = args.threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[args.backend]

    print(f"입력: {video_path}")
    print(f"백엔드: {args.backend} (threshold={threshold})")

    if args.backend == "pyscenedetect":
        scenes = detect_pyscenedetect(
            video_path, args.detector, threshold, args.downscale, args.min_scene_len
        )
    else:
        scenes = detect_ffmpeg_gpu(video_path, threshold, args.min_scene_len)

    fps, duration = probe_video(video_path)
    meta = {
        "video": str(Path(video_path).resolve()),
        "backend": args.backend,
        "detector": args.detector if args.backend == "pyscenedetect" else None,
        "threshold": threshold,
        "fps": round(fps, 3),
        "duration_seconds": round(duration, 3),
    }

    outdir = Path(args.output_dir)
    write_outputs(scenes, meta, outdir)

    num_cuts = max(len(scenes) - 1, 0)
    print(f"\n탐지된 장면: {len(scenes)}개  (컷 {num_cuts}개)")
    for s in scenes[:10]:
        print(f"  #{s['scene']:>3}  {s['start_timecode']} -> {s['end_timecode']}")
    if len(scenes) > 10:
        print(f"  ... (총 {len(scenes)}개, 전체는 scenes.csv 참고)")
    print(f"\n출력: {outdir / 'scenes.csv'} , {outdir / 'cuts.json'}")

    if args.save_images:
        save_images(scenes, video_path, outdir)
    if args.split:
        split_clips(scenes, video_path, outdir, args.gpu_encode)


if __name__ == "__main__":
    main()
