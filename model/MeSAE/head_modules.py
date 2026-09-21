"""Swappable pieces of the finetune head (ADR 0016). Parameter names (p, q) and their init
match the pre-refactor MeSAEFeatureHead so saved checkpoints load unchanged.

Shapes: a, b are the spatially mixed code amplitudes [B, N', K, S] (B trials, N' patches,
K spatial filters, S alive stamps); power = a^2 + b^2."""
import torch
import torch.nn as nn


def spatial_mix(spatial, t, dim):
    """Signed spatial filter (nn.Linear(C, K, bias=False)) over channel axis `dim`, or
    identity when spatial is None (channel concat)."""
    if spatial is None:
        return t
    return torch.movedim(spatial(torch.movedim(t, dim, -1).float()), -1, dim)


class FlatTimePool(nn.Module):
    """Uniform weights over patches (ADR 0014 C0)."""
    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return power.mean(1)


class LearnedTimePool(nn.Module):
    """Low-rank softmax time weights w[s, n] = softmax_n(sum_r p[r, s] q[r, n]) (ADR 0014 C1).
    Small init => starts equal to the flat mean."""
    def __init__(self, rank, num_stamps, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_stamps) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def weights(self):                                           # [S, N']
        return torch.softmax(torch.einsum('rs,rn->sn', self.p, self.q), dim=-1)

    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return torch.einsum('sn,bnks->bks', self.weights(), power)


class EvokedBranch(nn.Module):
    """Signed low-rank time filter T[s, n] = 1/N' + sum_r p[r, s] q[r, n] applied LINEARLY to
    a and b (phase-locked content survives a linear functional, not power) (ADR 0014 C4)."""
    def __init__(self, rank, num_stamps, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_stamps) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def forward(self, a, b):                                     # -> [B, K, 2*S]
        T = 1.0 / a.shape[1] + torch.einsum('rs,rn->sn', self.p, self.q)
        return torch.cat([torch.einsum('sn,bnks->bks', T, a),
                          torch.einsum('sn,bnks->bks', T, b)], dim=-1)


def phase_advance(a, b):
    """z[k, s] = sum_n u[n+1] conj(u[n]), u = a + i b, as real/imag parts (no complex dtype)
    (ADR 0014 C3). Returns cat([z_re, z_im], -1) [B, K, 2*S]."""
    a_next, a_prev, b_next, b_prev = a[:, 1:], a[:, :-1], b[:, 1:], b[:, :-1]
    z_re = (a_next * a_prev + b_next * b_prev).sum(1)
    z_im = (b_next * a_prev - a_next * b_prev).sum(1)
    return torch.cat([z_re, z_im], dim=-1)


if __name__ == '__main__':
    # self-check against independent formulas (complex dtype, explicit softmax)
    torch.manual_seed(0)
    B, N, K, S, R = 3, 39, 8, 25, 2
    a, b = torch.randn(B, N, K, S), torch.randn(B, N, K, S)
    u = torch.complex(a, b)
    z = (u[:, 1:] * u[:, :-1].conj()).sum(1)
    pa = phase_advance(a, b)
    assert torch.allclose(pa[..., :S], z.real, atol=1e-5) and torch.allclose(pa[..., S:], z.imag, atol=1e-5)
    ltp = LearnedTimePool(R, S, N)
    nn.init.normal_(ltp.p); nn.init.normal_(ltp.q)
    w = ltp.weights()
    assert torch.allclose(w.sum(-1), torch.ones(S), atol=1e-6)
    power = a.pow(2) + b.pow(2)
    ref = torch.einsum('sn,bnks->bks', w, power)
    assert torch.allclose(ltp(power), ref, atol=1e-6)
    nn.init.zeros_(ltp.p)                                        # zero logits => uniform => flat mean
    assert torch.allclose(ltp(power), FlatTimePool()(power), atol=1e-5)
    ev = EvokedBranch(R, S, N)
    nn.init.zeros_(ev.p)                                         # T = 1/N' => trial mean of a, b
    out = ev(a, b)
    assert torch.allclose(out[..., :S], a.mean(1), atol=1e-5) and torch.allclose(out[..., S:], b.mean(1), atol=1e-5)
    lin = nn.Linear(64, K, bias=False)
    t = torch.randn(B, N, 64, S)
    assert spatial_mix(lin, t, 2).shape == (B, N, K, S) and spatial_mix(None, t, 2) is t
    assert [n for n, _ in ltp.named_parameters()] == ['p', 'q'] and [n for n, _ in ev.named_parameters()] == ['p', 'q']
    print('head_modules self-check OK')
