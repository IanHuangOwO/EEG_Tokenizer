"""EEGNet -- a compact CNN trained from scratch per task; the specialist baseline every EEG paper reports.

    Lawhern VJ, Solon AJ, Waytowich NR, Gordon SM, Hung CP, Lance BJ. EEGNet: a compact convolutional neural network
    for EEG-based brain-computer interfaces. Journal of Neural Engineering 15(5):056013, 2018. arXiv 1611.08024.

Architecture as in EEG-FM-Compass (github.com/Dingkun0817/EEG-FM-Benchmark, models/DL/EEGNet/Model_EEGNet.py), unchanged,
so our numbers compare with their tables: EEGNet-8,2 (F1 8, D 2, F2 16, temporal kernel 64), dropout configurable.
A temporal convolution (band-pass-like filters) -> a depthwise spatial convolution per filter (a spatial pattern per
band, CSP-like) -> a separable convolution -> a linear classifier. Input [B, C, T] in microvolts: the BatchNorm eps
(1e-5) swamps a signal in volts (docs/reports/2026-10-06-eegnet-compass-check.md). Trained by train_baseline.py.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, n_classes, chans, samples, kern=64, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern // 2 - 1, kern - kern // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, kern), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),   # depthwise spatial
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),                       # separable: depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2),                       # + pointwise
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.classifier_block = nn.Sequential(nn.Linear(F2 * (samples // 32), n_classes))

    def forward(self, x):                                   # x [B, C, T] -> logits [B, n_classes]
        x = self.block2(self.block1(x.unsqueeze(1)))
        return self.classifier_block(x.flatten(1))


def _selfcheck():
    """Shape and size: BNCI2014001 at 250 Hz, 5 s (Compass loso input) -> 4 classes; EEGNet-8,2 is ~2-3k parameters."""
    m = EEGNet(4, 22, 1250)
    out = m(torch.randn(3, 22, 1250))
    assert out.shape == (3, 4), out.shape
    n = sum(p.numel() for p in m.parameters())
    assert 1500 < n < 5000, n
    print(f'EEGNet selfcheck ok ({n} parameters)')


if __name__ == '__main__':
    _selfcheck()
