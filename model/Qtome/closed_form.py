"""Closed-form finetune head (2026-10-04/05): fit without SGD on the frozen Q-atom code (a, b), for few-shot.

training_params.finetune.fit = "closed_form", closed_form = {"spatial_k", "atom_rank", "time_rank", "signed_time_rank"}:
the loso all-atom head (atom_power + signed_ab, Qtome_modules.FeatureHead) with every factor set in closed form instead
of by SGD, same ranks (user: few-shot keeps the head's structure so its factors read as Q-atom events, not a generic
probe):
  atom_power: spatial filter C -> spatial_k shared over Q-atoms and a/b (CSP on the trial covariance of the code,
    patches x Q-atoms x {a, b} as samples) -> a^2 + b^2 -> per-atom time weights w[s, n] (non-negative, rank
    time_rank: the leading SVD components of the per-(Q-atom, patch) Fisher score of log-power, clipped at 0, summed
    to 1 per Q-atom) -> log -> [K * S].
  signed_ab: spatial filter C -> spatial_k (generalised eigenvectors of the class-mean code covariance vs the trial
    covariance, xDAWN-style) -> Q-atom pooling S -> atom_rank and time filters N' -> time_rank, the leading
    eigenvectors of the class-mean difference scatter over Q-atoms and over patches (HOSVD of the centred class
    means) -> [2 * K * atom_rank * time_rank].
  both concatenated -> shrinkage LDA, equal priors.
Every step is a formula, an eigen-decomposition or one convex solve: the same training trials give the same head
(no seed, no epochs, no learning rate). Loses to the SGD head on loso (no cross-subject alignment): few-shot only.
Picked on DEV few-shot over the per-paradigm closed-form heads it replaced (CSP power, xDAWN tangent space, TRCA;
removed 2026-10-07, docs/reports/2026-10-06-structured-fewshot-head.md).
"""
import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ClosedFormHead:
    def __init__(self, spatial_k=8, atom_rank=4, time_rank=2, signed_time_rank=None):
        self.K, self.M, self.R = int(spatial_k), int(atom_rank), int(time_rank)
        # signed_ab's own time_rank (a per-entry key, as in the SGD head); 'full' = every patch kept. DEV few-shot
        # 2026-10-06 (MI / P300 / SSVEP power / SSVEP phase-locked, mean): rank 2 53.2, 4 54.9, 8 56.7, full 59.7.
        self.Rs = None if signed_time_rank == 'full' else int(signed_time_rank) if signed_time_rank else self.R

    @staticmethod
    def _top_geig(P, Q, k):
        """Leading k generalised eigenvectors of P w = l Q w (Q shrunk 1e-6 of its trace) -> [C, k]."""
        from scipy.linalg import eigh
        Q = Q + 1e-6 * np.trace(Q) / len(Q) * np.eye(len(Q))
        _, V = eigh(P, Q)
        return V[:, ::-1][:, :k]

    def fit(self, amp, y):
        amp, y = np.asarray(amp, dtype=np.float64), np.asarray(y)
        n, N, C, S, _ = amp.shape
        K = min(self.K, C)
        cls = np.unique(y)
        X = amp.transpose(0, 2, 1, 3, 4).reshape(n, C, -1)                        # [n, C, N*S*2]
        # atom_power spatial filter: CSP on the code covariance (multi-class by pyriemann's AJD)
        covs = Covariances('oas').fit_transform(X)
        csp = CSP(nfilter=K, metric='euclid', log=False).fit(covs, y)
        self.Wp = csp.filters_[:K].T                                              # [C, K]
        p = np.einsum('ntcsj,ck->ntksj', amp, self.Wp)
        p = (p ** 2).sum(-1)                                                      # [n, N, K, S]
        lp = np.log(p + 1e-12)
        mu = np.stack([lp[y == c].mean(0) for c in cls])                          # [k, N, K, S]
        within = sum(((lp[y == c] - mu[i]) ** 2).sum(0) for i, c in enumerate(cls)) / max(n - len(cls), 1)
        fisher = (mu.var(0) / (within + 1e-12)).mean(1)                           # [N, S]
        U, sv, Vt = np.linalg.svd(fisher.T, full_matrices=False)                  # [S, N] rank-R
        w = np.clip((U[:, :self.R] * sv[:self.R]) @ Vt[:self.R], 0, None) + 1e-6
        self.wt = w / w.sum(1, keepdims=True)                                     # [S, N]
        # signed_ab spatial filter: class-mean code covariance vs trial covariance
        Xc = X - X.mean(-1, keepdims=True)
        means = np.stack([Xc[y == c].mean(0) for c in cls])                       # [k, C, T]
        Q = np.einsum('nct,ndt->cd', Xc, Xc) / n
        # one generalised eigenproblem over all class means ('pooled'). DEV 2026-10-05: per-class filters (xDAWN
        # proper) +2.4 on P300 but -0.7 MI / -2.6 SSVEP; noise-scaled (Fisher) factors -0.6..-1.6: both dropped.
        self.Ws = self._top_geig(sum(m @ m.T for m in means) / len(cls), Q, K)    # [C, K]
        ys = np.einsum('ntcsj,ck->nkjst', amp, self.Ws)                           # [n, K, 2, S, N]
        cm = np.stack([ys[y == c].mean(0) for c in cls])
        D = (cm - cm.mean(0)).reshape(-1, S, N)                                   # centred class means
        self.Us = np.linalg.eigh(np.einsum('isn,itn->st', D, D))[1][:, ::-1][:, :self.M]     # [S, M]
        self.Qt = np.linalg.eigh(np.einsum('isn,ism->nm', D, D))[1][:, ::-1][:, :self.Rs]    # [N, Rs]
        k = len(cls)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.full(k, 1 / k))   # logreg readout: -1..-2 (DEV, 2026-10-05)
        self.clf = make_pipeline(StandardScaler(), lda).fit(self._structured_features(amp), y)
        return self

    def _structured_features(self, amp):
        p = (np.einsum('ntcsj,ck->ntksj', amp, self.Wp) ** 2).sum(-1)              # [n, N, K, S]
        fp = np.log(np.einsum('sn,bnks->bks', self.wt, p) + 1e-12).reshape(len(amp), -1)        # [n, K*S]
        ys = np.einsum('ntcsj,ck->nkjst', amp, self.Ws)
        fs = np.einsum('nkjst,sm,tr->nkjmr', ys, self.Us, self.Qt).reshape(len(amp), -1)
        return np.concatenate([fp, fs], 1)

    def predict(self, amp):
        return self.clf.predict(self._structured_features(np.asarray(amp, dtype=np.float64)))


def _selfcheck():
    """Separable synthetic data: class 1 has extra power on channel 0 in Q-atom 0 and a fixed signed offset on
    channel 1; held-out trials must classify well above chance."""
    rng = np.random.default_rng(0)
    n, P, C, S = 80, 6, 4, 3
    y = np.repeat([0, 1], n // 2)
    amp = rng.standard_normal((n, P, C, S, 2))
    amp[y == 1, :, 0, 0, :] *= 3.0
    amp[y == 1, :, 1, :, 0] += 1.0
    tr, ev = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    acc = (ClosedFormHead(spatial_k=2).fit(amp[tr], y[tr]).predict(amp[ev]) == y[ev]).mean()
    assert acc > 0.8, acc
    print('closed_form selfcheck ok')


if __name__ == '__main__':
    _selfcheck()
