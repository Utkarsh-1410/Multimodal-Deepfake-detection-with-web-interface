import random
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from data_utils.utils import *
from utils import *
from PIL import Image
from glob import glob

class SimpleImageFolder(Dataset):
    def __init__(self, root, transforms_=None):
        self.root = root
        all_files = glob(root + "/*")
        self.data_list = [os.path.abspath(f) for f in all_files]
        self.data_len = len(self.data_list)
        self.transforms = transforms_

    def __getitem__(self, index):
        img_name = self.data_list[index]
        img = Image.open(self.data_list[index])
        if self.transforms:
            img = self.transforms(img)
        return img, img_name

    def __len__(self) -> int:
        return self.data_len


class SequentialFramesDataset(Dataset):
    """Loads frames in filename-sorted order as a single sequence per video directory.
    Returns a tensor stack (T, C, H, W) and list of frame names.
    """

    def __init__(self, root, transforms_=None, max_frames=None):
        self.root = root
        files = sorted(glob(root + "/*"))
        self.data_list = [os.path.abspath(f) for f in files]
        self.transforms = transforms_
        self.max_frames = max_frames

    def __getitem__(self, index):
        # For simplicity, this dataset yields one item: the entire ordered sequence
        # Therefore, ignore index and return the full sequence
        frame_paths = self.data_list
        if self.max_frames is not None and len(frame_paths) > self.max_frames:
            frame_paths = frame_paths[: self.max_frames]
        images = []
        for fp in frame_paths:
            img = Image.open(fp)
            if self.transforms:
                img = self.transforms(img)
            images.append(img)
        if len(images) == 0:
            return None, []
        # Stack to (T, C, H, W)
        seq = torch.stack(images, dim=0)
        return seq, frame_paths

    def __len__(self) -> int:
        # Single sequence per directory for app usage
        return 1
