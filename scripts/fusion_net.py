import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusionNet(nn.Module):
    """
    Tiny network to fuse DepthSplat and DIFFIX images.

    Input:  tensor of shape (B, 6, H, W)
            channels 0-2: DepthSplat RGB
            channels 3-5: DIFFIX RGB
    Output: tensor of shape (B, 3, H, W) in [0, 1]
    """

    def __init__(self):
        super().__init__()
        # 6 -> 16 -> 3 is already enough for local mixing
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))  # keep output in [0, 1]
        return x


def load_fusion_model(weights_path: str, map_location=None) -> SimpleFusionNet:
    """Load a trained SimpleFusionNet from a .pt or .pth file."""
    if map_location is None:
        map_location = "cpu"
    model = SimpleFusionNet()
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()
    return model
