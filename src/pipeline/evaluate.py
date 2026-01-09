from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from rich import print

from src.config import cfg, Config
from src.pipeline.inference import RealtimeInference


@dataclass
class VideoSample:
    path: Path
    label: int  # 0 real, 1 fake


def list_videos(root: Path) -> List[VideoSample]:
    samples: List[VideoSample] = []
    for label_name, label in [("real", 0), ("fake", 1)]:
        class_dir = root / label_name
        if not class_dir.exists():
            continue
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            for p in class_dir.rglob(f"*{ext}"):
                samples.append(VideoSample(path=p, label=label))
    return samples


def pick_balanced(samples: List[VideoSample], max_videos: int) -> List[VideoSample]:
    if max_videos <= 0:
        return samples
    half = max_videos // 2
    real = [s for s in samples if s.label == 0][:half]
    fake = [s for s in samples if s.label == 1][:max_videos - len(real)]
    if len(fake) < half:
        # try to top-up from remaining of either class
        remaining = [s for s in samples if s not in real + fake][: max_videos - (len(real) + len(fake))]
        fake += remaining
    return real + fake


def predict_video(engine: RealtimeInference, video_path: Path, frame_stride: int = 2, fast_no_face: bool = False) -> float:
    # Run the pipeline over the video and aggregate frame scores
    from src.data.capture import VideoSource, stream_frames
    from src.data.preprocessing import FacePreprocessor
    from src.features.vision_features import compute_region_attention_mask

    pre = FacePreprocessor(img_size=engine.cfg.infer.face_img_size, max_faces=1)
    scores: List[float] = []
    for idx, frame in stream_frames(VideoSource("file", str(video_path), frame_stride)):
        if fast_no_face:
            h, w = frame.shape[:2]
            size = min(h, w)
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            crop = frame[y0:y0+size, x0:x0+size]
            aligned = cv2.resize(crop, (engine.cfg.infer.face_img_size, engine.cfg.infer.face_img_size))
            mask = np.ones((engine.cfg.infer.face_img_size, engine.cfg.infer.face_img_size), dtype=np.float32)
            prob = engine._prob_fake(aligned, mask)
        else:
            faces = pre.detect_and_align(frame)
            if not faces:
                continue
            f = faces[0]
            mask = compute_region_attention_mask(f.aligned, f.landmarks)
            prob = engine._prob_fake(f.aligned, mask)
        scores.append(prob)
    if not scores:
        return 0.5  # uncertain
    return float(np.median(scores))


def evaluate(
    cfg: Config,
    max_videos: int | None = None,
    frame_stride: int = 2,
    fast_no_face: bool = False,
    threshold: float | None = None,
    balanced: bool = False,
    sweep: bool = False,
) -> None:
    test_root = cfg.paths.datasets_root / "test"
    samples = list_videos(test_root)
    if not samples:
        print(f"[yellow]No videos found under {test_root}[/yellow]")
        return
    if max_videos is not None and max_videos > 0:
        samples = pick_balanced(samples, max_videos) if balanced else samples[: max_videos]
    # Ensure non-visual run
    cfg.infer.show_visuals = False
    engine = RealtimeInference(cfg)

    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    for i, s in enumerate(samples, 1):
        score = predict_video(engine, s.path, frame_stride=frame_stride, fast_no_face=fast_no_face)
        pred = int(score >= cfg.infer.detection_threshold)
        y_true.append(s.label)
        y_pred.append(pred)
        y_score.append(score)
        print(f"[{i}/{len(samples)}] {s.path.name}: score={score:.3f} pred={pred} label={s.label}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = float(tp / (tp + fp + 1e-6))
    rec = float(tp / (tp + fn + 1e-6))
    print(f"[bold]Accuracy[/bold]: {acc:.3f}")
    print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  TP:{tp} FP:{fp} TN:{tn} FN:{fn}")

    if threshold is not None:
        thr = float(threshold)
        y_pred_t = (np.array(y_score) >= thr).astype(int)
        tp = int(((y_pred_t == 1) & (y_true == 1)).sum())
        tn = int(((y_pred_t == 0) & (y_true == 0)).sum())
        fp = int(((y_pred_t == 1) & (y_true == 0)).sum())
        fn = int(((y_pred_t == 0) & (y_true == 1)).sum())
        acc_t = float((y_true == y_pred_t).mean())
        prec_t = float(tp / (tp + fp + 1e-6))
        rec_t = float(tp / (tp + fn + 1e-6))
        print(f"With threshold={thr:.2f} → Acc={acc_t:.3f} Prec={prec_t:.3f} Rec={rec_t:.3f}")

    if sweep:
        print("Sweep thresholds:")
        for thr in np.linspace(0.3, 0.9, 13):
            y_pred_t = (np.array(y_score) >= thr).astype(int)
            tp = int(((y_pred_t == 1) & (y_true == 1)).sum())
            tn = int(((y_pred_t == 0) & (y_true == 0)).sum())
            fp = int(((y_pred_t == 1) & (y_true == 0)).sum())
            fn = int(((y_pred_t == 0) & (y_true == 1)).sum())
            acc_t = float((y_true == y_pred_t).mean())
            prec_t = float(tp / (tp + fp + 1e-6))
            rec_t = float(tp / (tp + fn + 1e-6))
            print(f"thr={thr:.2f} acc={acc_t:.3f} prec={prec_t:.3f} rec={rec_t:.3f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--frame_stride", type=int, default=2)
    p.add_argument("--fast_no_face", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--sweep", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        cfg,
        max_videos=(args.max_videos if args.max_videos > 0 else None),
        frame_stride=args.frame_stride,
        fast_no_face=args.fast_no_face,
        threshold=args.threshold,
        balanced=bool(args.balanced),
        sweep=bool(args.sweep),
    )


