from deep_fake_detect.utils import *
import torch
import torch.nn as nn
from utils import *
from data_utils.utils import *
import torch.nn as nn
import torch.nn.functional as F
from deep_fake_detect.features import *


class DeepFakeDetectModel(nn.Module):
    """
    This is simple model which takes in each frame of video independently and classified them.
    Later the entire video is classified based upon heuristics, which is not done by this model.
    For the frame passed, features are extracted, using given encoder. Then applies AdaptiveAvgPool2d, flattens the
    features and passes to classifier.

    cnn_encoder:
      default: ['tf_efficientnet_b0_ns', 'tf_efficientnet_b7_ns'] # choose either one
    training:
      train_size: 1
      valid_size: 1
      test_size: 1
      params:
        model_name: 'DeepFakeDetectModel'
        label_smoothing: 0.1 # 0 to disable this, or any value less than 1
        train_transform: ['simple', 'complex'] # choose either of the data augmentation
        batch_format: 'simple' # Do not change
        # Adjust epochs, learning_rate, batch_size , fp16, opt_level
        epochs: 5
        learning_rate: 0.001
        batch_size: 4
        fp16: True
        opt_level: 'O1'
        dataset: ['optical', 'plain'] # choose either of the data type
    """

    def __init__(self, frame_dim=None, encoder_name=None):
        super().__init__()

        self.image_dim = frame_dim
        self.num_of_classes = 1
        self.encoder = get_encoder(encoder_name)
        self.encoder_flat_feature_dim, _ = get_encoder_params(encoder_name)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_flat_feature_dim, int(self.encoder_flat_feature_dim * .10)),
            nn.Dropout(0.50),
            nn.ReLU(),
            nn.Linear(int(self.encoder_flat_feature_dim * .10), self.num_of_classes),
        )

    def forward(self, x):
        # x shape = batch_size x color_channels x image_h x image_w
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.classifier(x)
        return x


class TemporalHead(nn.Module):
    """Lightweight temporal head over per-frame embeddings.
    Input: (B, T, D) frame embeddings
    Output: (B, 1) logit
    """

    def __init__(self, embedding_dim: int, hidden_ratio: float = 0.25, kernel_size: int = 3):
        super().__init__()
        hidden_dim = max(8, int(embedding_dim * hidden_ratio))
        self.proj = nn.Linear(embedding_dim, hidden_dim)
        self.temporal = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        b, t, d = x.shape
        x = self.proj(x)              # (B, T, H)
        x = x.transpose(1, 2)         # (B, H, T)
        x = self.temporal(x)
        x = self.act(x)
        x = self.pool(x).squeeze(-1)  # (B, H)
        x = self.classifier(x)        # (B, 1)
        return x