from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    label: int  # 0 real, 1 fake


class ReplayBuffer:
    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self.data: Deque[Sample] = deque(maxlen=capacity)

    def add(self, image: np.ndarray, label: int) -> None:
        self.data.append(Sample(image=image, label=label))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        import random

        batch = random.sample(self.data, k=min(batch_size, len(self.data)))
        imgs = np.stack([b.image for b in batch])
        labels = np.array([b.label for b in batch], dtype=np.int64)
        return imgs, labels

    def __len__(self) -> int:
        return len(self.data)


