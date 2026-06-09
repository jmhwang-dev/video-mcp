# video-scene-analysis

유튜브 영상을 받아(영상 전용, video-only) 저장하고, 그 영상에서 **장면 전환 지점
(shot boundary)** 을 탐지하는 도구입니다. 환경은 [uv](https://docs.astral.sh/uv/) 로 관리합니다.

## 구성

| 파일 | 설명 |
|---|---|
| `download_video.py` | 유튜브 URL → 해상도를 대화형으로 골라 **video-only** 다운로드 |
| `detect_scenes.py` | 장면 전환 탐지 (CPU / GPU 백엔드), `scenes.csv`·`cuts.json` 출력 |
| `scene_segmentation.md` | 장면 분할 방법론 (ffmpeg / PySceneDetect / TransNetV2 비교, CPU·GPU 가이드) |

## 사전 요구사항

- `uv`, `ffmpeg`(시스템 설치) — GPU 가속을 쓰려면 NVIDIA GPU + CUDA 지원 ffmpeg.

## 설치

```bash
uv sync
```

## 1) 영상 다운로드 (video-only)

```bash
# 대화형 해상도 메뉴
uv run download_video.py "https://www.youtube.com/watch?v=XXXX"

# 메뉴 없이 해상도 지정
uv run download_video.py "https://www.youtube.com/watch?v=XXXX" --height 720

# 사용 가능한 해상도만 확인
uv run download_video.py "https://www.youtube.com/watch?v=XXXX" --list
```

영상은 현재 폴더에 `제목 [영상ID] [높이p].확장자` 형식으로 저장됩니다.
장면 탐지용이라 **오디오는 받지 않습니다.**

## 2) 장면 전환 탐지

```bash
# 기본: PySceneDetect (CPU, 안정적)
uv run detect_scenes.py "받은영상.mp4"

# GPU(NVDEC) 디코딩으로 빠르게 (ffmpeg scene 필터)
uv run detect_scenes.py "받은영상.mp4" --backend ffmpeg-gpu --threshold 0.4

# 빠른 모션 영상: adaptive 디텍터 + 대표 프레임 저장 + 구간 분할
uv run detect_scenes.py "받은영상.mp4" --detector adaptive --save-images --split
```

주요 옵션:

- `--backend pyscenedetect|ffmpeg-gpu` — 탐지 백엔드 (기본 `pyscenedetect`)
- `--detector content|adaptive` — PySceneDetect 디텍터
- `--threshold FLOAT` — 임계값 (기본 27.0 / ffmpeg-gpu 는 0.4)
- `--downscale INT` — 속도용 다운스케일 (pyscenedetect)
- `--min-scene-len SEC` — 최소 장면 길이(초)
- `--save-images` — 장면별 대표 프레임 저장 (`scenes_images/`)
- `--split` — 구간별 클립 분할 (`clips/`), `--gpu-encode` 시 `h264_nvenc` 사용

### 출력

- `scenes.csv` — 장면별 시작/끝 타임코드·프레임
- `cuts.json` — 컷 지점(`cut_points_seconds`) + 장면 리스트 + 메타데이터

자세한 방법론과 CPU/GPU 선택 기준은 [`scene_segmentation.md`](./scene_segmentation.md) 참고.
