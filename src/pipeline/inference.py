from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.data.capture import VideoSource, stream_frames
from src.data.preprocessing import FacePreprocessor
from src.features.vision_features import compute_region_attention_mask, compute_texture_stats
from src.models.discriminator import SmallAttentionDiscriminator
from src.models.generator import SimplePerturbationGenerator
from src.pipeline.decisions import make_decision


@dataclass
class FrameResult:
    index: int
    prob_fake: float
    bbox: Optional[tuple[int, int, int, int]]


class RealtimeInference:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.infer.device if torch.cuda.is_available() else "cpu")
        self.pre = FacePreprocessor(img_size=cfg.infer.face_img_size, max_faces=cfg.infer.max_faces)
        self.gen = SimplePerturbationGenerator().to(self.device)
        self.disc = SmallAttentionDiscriminator().to(self.device)
        # Try to load trained weights if available
        try:
            ckpt_path = (cfg.paths.models_dir / "best.pt").resolve()
            if ckpt_path.exists():
                state = torch.load(str(ckpt_path), map_location=self.device)
                if isinstance(state, dict):
                    if "disc" in state:
                        self.disc.load_state_dict(state["disc"], strict=False)
                    if "gen" in state:
                        self.gen.load_state_dict(state["gen"], strict=False)
        except Exception:
            pass
        self.disc.eval()

    def _prior_from_mask(self, mask: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(mask).float().to(self.device)
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return t

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        bgr = img[:, :, ::-1].copy()
        t = torch.from_numpy(bgr.transpose(2, 0, 1)).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def _prob_fake(self, face_img: np.ndarray, mask: np.ndarray) -> float:
        x = self._to_tensor(face_img)
        x = self.gen(x)
        prior = F.interpolate(self._prior_from_mask(mask), size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.disc(x, prior=prior)
        prob = torch.sigmoid(logits).item()
        return float(prob)

    def _draw(self, frame: np.ndarray, res: Optional[FrameResult]) -> np.ndarray:
        vis = frame.copy()
        if res and res.bbox:
            x0, y0, x1, y1 = res.bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255) if res.prob_fake >= self.cfg.infer.detection_threshold else (0, 255, 0), 2)
            cv2.putText(vis, f"fake: {res.prob_fake:.2f}", (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return vis

    def run(self) -> None:
        video = VideoSource(self.cfg.infer.source, self.cfg.infer.video_path, self.cfg.infer.frame_stride)
        last_vis = None
        for idx, frame in stream_frames(video):
            faces = self.pre.detect_and_align(frame)
            result: Optional[FrameResult] = None
            if faces:
                f = faces[0]
                mask = compute_region_attention_mask(f.aligned, f.landmarks)
                _ = compute_texture_stats(f.aligned)  # placeholder; could be fused later
                prob = self._prob_fake(f.aligned, mask)
                decision = make_decision(prob, self.cfg.infer.detection_threshold)
                result = FrameResult(index=idx, prob_fake=decision.score, bbox=f.bbox)
            if self.cfg.infer.show_visuals:
                vis = self._draw(frame, result)
                cv2.imshow("Deepfake Detection", vis)
                last_vis = vis
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        if self.cfg.infer.show_visuals:
            if last_vis is not None:
                cv2.imshow("Deepfake Detection", last_vis)
            cv2.destroyAllWindows()


