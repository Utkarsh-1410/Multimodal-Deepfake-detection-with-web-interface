from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class Paths:
    # Default to FaceForensics under project data; override with DATASETS_ROOT env var
    datasets_root: Path = Path(os.getenv("DATASETS_ROOT", "FaceForensics"))
    logs_dir: Path = Path("runs")
    models_dir: Path = Path("checkpoints")
    export_dir: Path = Path("exports")


@dataclass
class InferenceConfig:
    source: str = "webcam"  # "webcam" | "file"
    video_path: str | None = None
    audio_path: str | None = None
    device: str = "cuda"  # "cuda" | "cpu"
    detection_threshold: float = 0.5
    frame_stride: int = 2  # sample every N frames
    face_img_size: int = 224
    max_faces: int = 1
    show_visuals: bool = True


@dataclass
class OptimizeConfig:
    dynamic_quantization: bool = True
    prune_amount: float = 0.2
    onnx_opset: int = 17


@dataclass
class TrainingConfig:
    batch_size: int = 16
    lr: float = 2e-4
    epochs: int = 10
    num_workers: int = 4


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)


cfg = Config()


