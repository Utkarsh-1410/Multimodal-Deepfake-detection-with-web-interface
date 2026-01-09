import argparse
from rich import print

from src.config import cfg
from src.pipeline.inference import RealtimeInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Deepfake Detection")
    parser.add_argument("--source", choices=["webcam", "file"], default=cfg.infer.source)
    parser.add_argument("--video_path", type=str, default=cfg.infer.video_path)
    parser.add_argument("--audio_path", type=str, default=cfg.infer.audio_path)
    parser.add_argument("--device", type=str, default=cfg.infer.device)
    parser.add_argument("--threshold", type=float, default=cfg.infer.detection_threshold)
    parser.add_argument("--no_viz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg.infer.source = args.source
    cfg.infer.video_path = args.video_path
    cfg.infer.audio_path = args.audio_path
    cfg.infer.device = args.device
    cfg.infer.detection_threshold = args.threshold
    cfg.infer.show_visuals = not args.no_viz

    print(f"[bold cyan]Launching pipeline[/bold cyan]: {cfg.infer}")
    engine = RealtimeInference(cfg)
    engine.run()


if __name__ == "__main__":
    main()


