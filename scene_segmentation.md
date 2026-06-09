# 영상 구간 분할 (장면 전환 탐지) 방법론

받은 유튜브 영상에서 **장면 전환 지점(shot boundary)** 을 찾아 영상을 구간으로 나누는
방법을 정리한 문서입니다. 이 저장소의 `detect_scenes.py` 가 아래 A·B 방법을 실제로
구현하며, C(ML)는 정확도 업그레이드 경로로 안내합니다.

## 0. 용어

- **하드 컷(hard cut)**: 한 프레임에서 다음 프레임으로 장면이 즉시 바뀜. 탐지 쉬움.
- **점진 전환(gradual transition)**: 디졸브/페이드/와이프 등 여러 프레임에 걸쳐 전환.
  단순 임계값 방식이 가장 놓치기 쉬운 케이스.
- **장면(scene) vs 샷(shot)**: 여기서는 카메라/구도가 바뀌는 *샷 경계* 를 기준으로
  구간을 나눕니다.

## 1. 방법 비교

### A. ffmpeg `scene` 필터 (임계값 기반) — GPU 디코딩 가능
연속 프레임 간 차이를 0~1 점수로 계산해 임계값을 넘으면 컷으로 판단합니다.

```bash
# NVDEC(GPU)로 디코딩 + 장면 점수 > 0.4 인 프레임의 타임스탬프 출력
ffmpeg -hwaccel cuda -i input.mp4 \
  -vf "select='gt(scene,0.4)',showinfo" -an -f null -
```

- **장점**: 추가 파이썬 의존성 없음, `-hwaccel cuda` 로 디코딩을 GPU(NVDEC)에 오프로드 →
  길고 고해상도인 영상에서 빠름.
- **단점**: 단일 임계값이라 장면 종류에 민감, 점진 전환에 약함, 장면 리스트/통계 관리 기능 없음.
- **본 저장소**: `--backend ffmpeg-gpu` (기본 threshold 0.4, GPU 실패 시 CPU 디코딩으로 자동 폴백).

### B. PySceneDetect — 안정적인 기본 (주로 CPU)
HSV 색/콘텐츠 변화량을 분석하는 잘 검증된 라이브러리입니다.

- **ContentDetector**: 콘텐츠 변화 기반(기본). 하드 컷에 강함. (threshold 기본 27.0)
- **AdaptiveDetector**: 이동 평균 대비 상대 변화로 판단 → 카메라 흔들림/빠른 모션에서 오탐 감소.
- **ThresholdDetector**: 밝기 기반(페이드 인/아웃 탐지에 유용).
- 통계 CSV, 대표 프레임 저장, 구간 분할(split) 등 부가 기능 내장. `--downscale` 로 가속.
- 디코딩 자체는 주로 CPU(OpenCV)지만 짧은/저해상도 영상에는 충분.
- **본 저장소**: `--backend pyscenedetect --detector content|adaptive`.

### C. ML 기반 — TransNetV2 (최고 정확도, GPU 필수)
딥러닝 샷 경계 탐지 모델로, **점진 전환까지 포함**해 현존 최고 수준의 정확도를 냅니다.

- **장점**: 디졸브/페이드 등 어려운 전환도 잘 잡음, 다양한 콘텐츠에 견고.
- **단점**: PyTorch/TensorFlow + 모델 가중치 등 셋업이 무거움, GPU 추론 사실상 필수.
- **권장 시점**: A·B로 정확도가 부족할 때 도입. (현재 기본 의존성에서는 제외 — 필요 시
  `uv add torch` 후 TransNetV2 추론 코드를 추가.)

## 2. CPU vs GPU 가이드 (이 머신: RTX 6000 Ada, 49GB / ffmpeg CUDA·NVDEC 지원)

| 상황 | 권장 | 이유 |
|---|---|---|
| 짧은/저해상도 클립 | **B. PySceneDetect (CPU)** | 셋업 간단, 정확, GPU 불필요 |
| 길거나 1080p+ 영상 | **A. ffmpeg-gpu** 또는 B + `--downscale` | NVDEC 디코딩으로 처리량↑ |
| 점진 전환이 많은 영상 / 최고 정확도 | **C. TransNetV2 (GPU)** | 점진 전환까지 탐지 |

- NVDEC 디코딩(h264/hevc/vp9/av1 `*_cuvid`)은 **모든 방법의 디코딩 단계**를 가속합니다.
  특히 고해상도·장시간 영상에서 효과가 큽니다.
- ML 방법(C)은 추론 자체에 GPU가 필요하며, 이 머신의 GPU 메모리(49GB)면 충분합니다.
- 분할(split) 시 재인코딩이 필요하면 `h264_nvenc` 로 GPU 인코딩할 수 있습니다
  (`detect_scenes.py --split --gpu-encode`). 단순 분할은 무손실 stream copy가 기본입니다.

## 3. 출력 포맷 (A·B 공통)

`detect_scenes.py` 는 백엔드와 무관하게 동일한 산출물을 만듭니다.

- **`scenes.csv`**: 장면별 한 행 — `scene, start_timecode, start_seconds, start_frame,
  end_timecode, end_seconds, end_frame`
- **`cuts.json`**: 메타데이터(영상 경로/백엔드/threshold/fps/길이) + `cut_points_seconds`
  (장면 경계 타임스탬프) + 전체 `scenes` 리스트
- 옵션: `scenes_images/`(대표 프레임), `clips/`(구간별 클립)

## 4. 권장 워크플로우

1. `download_video.py` 로 영상을 video-only 로 받는다(오디오는 탐지에 불필요).
2. 우선 **B (PySceneDetect, CPU)** 로 빠르게 기준선을 잡는다.
3. 영상이 길거나 고해상도면 **A (ffmpeg-gpu)** 로 속도를 높이고 threshold를 조정한다.
4. 점진 전환 누락 등 정확도가 부족하면 **C (TransNetV2, GPU)** 로 업그레이드한다.
5. `cuts.json` 의 `cut_points_seconds` 를 후속 분석/편집 파이프라인의 입력으로 사용한다.
