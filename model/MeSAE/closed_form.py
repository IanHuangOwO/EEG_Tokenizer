"""Closed-form finetune head (2026-10-04): fit without SGD on the frozen Q-atom code (a, b), for few-shot.

training_params.finetune.fit = "closed_form", closed_form = {"branch": ..., "nfilter": 4}:
  power  -- induced power (MI): per Q-atom, the channel covariance of its a and b over the trial's patches (OAS
            shrinkage) -> CSP log-variance (nfilter filters) -> all Q-atoms concatenated -> shrinkage LDA.
            The current head's stamp_power is the same form (power of a spatially filtered code) with SGD filters.
  signed -- phase-locked (P300): signed a, b per channel (Q-atoms x patches as the time axis) -> xDAWN covariances
            (nfilter per class, OAS) -> tangent space -> logistic regression, L2 strength by 3-fold CV.
  trca   -- phase-locked, many classes (SSVEP, 2026-10-05): per Q-atom, ensemble TRCA (Nakanishi 2018) on the signed
            a, b per channel (patches as the time axis): per class the nfilter spatial filters that maximise the
            covariance between that class's trials; a trial's score for a class is the correlation of its filtered
            code with the filtered class mean, summed over Q-atoms (the Q-atoms play TRCA's filter-bank role); argmax.
  structured -- the loso all-atom head (stamp_power + signed_ab, MeSAE_modules.FeatureHead) with every factor
            set in closed form instead of by SGD, same ranks (2026-10-05, user: few-shot keeps the head's structure so
            its factors read as Q-atom events, not a generic probe):
              stamp_power: spatial filter C -> spatial_k shared over Q-atoms and a/b (CSP on the trial covariance of
                the code, patches x Q-atoms x {a, b} as samples) -> a^2 + b^2 -> per-atom time weights w[s, n]
                (non-negative, rank time_rank: the leading SVD components of the per-(Q-atom, patch) Fisher score of
                log-power, clipped at 0, summed to 1 per Q-atom) -> log -> [K * S].
              signed_ab: spatial filter C -> spatial_k (generalised eigenvectors of the class-mean code covariance vs
                the trial covariance, xDAWN-style) -> Q-atom pooling S -> stamp_rank and time filters N' -> time_rank,
                the leading eigenvectors of the class-mean difference scatter over Q-atoms and over patches
                (HOSVD of the centred class means) -> [2 * K * stamp_rank * time_rank].
              both concatenated -> shrinkage LDA, equal priors.
Every step is a formula, an eigen-decomposition or one convex solve: the same training trials give the same head
(no seed, no epochs, no learning rate). On DEV few-shot it beat the SGD head by +7.5 (BNCI2015001) and +5.4
(BNCI2014009) and lost to it on loso (no cross-subject alignment): few-shot only. Equal class priors / balanced class weights (P300 is 1:5).
"""
import numpy as np
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BRANCHES = ('power', 'signed', 'trca', 'structured')


class ClosedFormHead:
    def __init__(self, branch, nfilter=4, spatial_k=8, stamp_rank=4, time_rank=2, signed_time_rank=None):
        if branch not in BRANCHES:
            raise ValueError(f"closed_form.branch must be one of {BRANCHES}, got {branch!r}")
        self.branch, self.nfilter = branch, int(nfilter)
        self.K, self.M, self.R = int(spatial_k), int(stamp_rank), int(time_rank)
        # signed_ab's own time_rank (a per-entry key, as in the SGD head); 'full' = every patch kept. DEV few-shot
        # 2026-10-06 (MI / P300 / SSVEP power / SSVEP phase-locked, mean): rank 2 53.2, 4 54.9, 8 56.7, full 59.7.
        self.Rs = None if signed_time_rank == 'full' else int(signed_time_rank) if signed_time_rank else self.R

    # ---- structured: the all-atom head's factors in closed form ----
    @staticmethod
    def _top_geig(P, Q, k):
        """Leading k generalised eigenvectors of P w = l Q w (Q shrunk 1e-6 of its trace) -> [C, k]."""
        from scipy.linalg import eigh
        Q = Q + 1e-6 * np.trace(Q) / len(Q) * np.eye(len(Q))
        _, V = eigh(P, Q)
        return V[:, ::-1][:, :k]

    def _fit_structured(self, amp, y):
        n, N, C, S, _ = amp.shape
        K = min(self.K, C)
        cls = np.unique(y)
        X = amp.transpose(0, 2, 1, 3, 4).reshape(n, C, -1)                        # [n, C, N*S*2]
        # stamp_power spatial filter: CSP on the code covariance (multi-class by pyriemann's AJD)
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

    def _structured_features(self, amp):
        p = (np.einsum('ntcsj,ck->ntksj', amp, self.Wp) ** 2).sum(-1)              # [n, N, K, S]
        fp = np.log(np.einsum('sn,bnks->bks', self.wt, p) + 1e-12).reshape(len(amp), -1)        # [n, K*S]
        ys = np.einsum('ntcsj,ck->nkjst', amp, self.Ws)
        fs = np.einsum('nkjst,sm,tr->nkjmr', ys, self.Us, self.Qt).reshape(len(amp), -1)
        return np.concatenate([fp, fs], 1)

    def _views(self, amp):
        """amp [n, N', C, S, 2] -> power: [n, S, C, 2N'] (a and b over patches, per Q-atom);
        signed: [n, C, S*2*N'] (the signed code of every Q-atom as one long time axis)."""
        n, P, C, S, _ = amp.shape
        if self.branch in ('power', 'trca'):
            return amp.transpose(0, 3, 2, 4, 1).reshape(n, S, C, 2 * P)
        return amp.transpose(0, 2, 3, 4, 1).reshape(n, C, S * 2 * P)

    @staticmethod
    def _trca_filters(Xk, nfilter):
        """Xk [m, C, T] trials of one class -> [C, nfilter]: top generalised eigenvectors of S w = l Q w, S the
        summed cross-trial covariance (sum over i != j of X_i X_j^T), Q the summed within-trial covariance."""
        Xk = Xk - Xk.mean(-1, keepdims=True)
        tot = Xk.sum(0)
        Q = np.einsum('mct,mdt->cd', Xk, Xk)
        S = tot @ tot.T - Q
        Q = Q + 1e-6 * np.trace(Q) / len(Q) * np.eye(len(Q))
        L = np.linalg.cholesky(Q)
        Li = np.linalg.inv(L)
        vals, vecs = np.linalg.eigh(Li @ S @ Li.T)
        return Li.T @ vecs[:, ::-1][:, :nfilter]

    def _trca_scores(self, X):
        """X [n, S, C, T] -> [n, K]: per Q-atom, the correlation of W^T x with W^T (class mean), summed over Q-atoms."""
        out = np.zeros((len(X), len(self.classes)))
        for s, (W, tmpl) in enumerate(self.blocks):
            z = np.einsum('ck,nct->nkt', W, X[:, s]).reshape(len(X), -1)
            z = z - z.mean(1, keepdims=True)
            z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
            out += z @ tmpl.T
        return out

    def _features(self, X):
        if self.branch == 'power':
            return np.concatenate([t.transform(X[:, s]) for s, t in enumerate(self.blocks)], 1)
        return self.blocks[0].transform(X)

    def fit(self, amp, y):
        X = self._views(np.asarray(amp, dtype=np.float64))
        k = len(np.unique(y))
        if self.branch == 'structured':
            amp = np.asarray(amp, dtype=np.float64)
            self._fit_structured(amp, np.asarray(y))
            k = len(np.unique(y))
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.full(k, 1 / k))   # logreg readout: -1..-2 (DEV, 2026-10-05)
            self.clf = make_pipeline(StandardScaler(), lda).fit(self._structured_features(amp), y)
            return self
        if self.branch == 'trca':
            y = np.asarray(y)
            self.classes = np.unique(y)
            self.blocks = []
            for s in range(X.shape[1]):
                W = np.concatenate([self._trca_filters(X[y == c, s], self.nfilter) for c in self.classes], 1)
                t = np.stack([np.einsum('ck,ct->kt', W, X[y == c, s].mean(0)).ravel() for c in self.classes])
                t = t - t.mean(1, keepdims=True)
                self.blocks.append((W, t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)))
            return self
        if self.branch == 'power':
            self.blocks = [make_pipeline(Covariances('oas'), CSP(nfilter=self.nfilter, metric='euclid', log=True))
                           .fit(X[:, s], y) for s in range(X.shape[1])]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=np.full(k, 1 / k))
        else:
            self.blocks = [make_pipeline(XdawnCovariances(nfilter=self.nfilter, estimator='oas'),
                                         TangentSpace(metric='riemann')).fit(X, y)]
            clf = LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=3, max_iter=2000, class_weight='balanced')
        self.clf = make_pipeline(StandardScaler(), clf).fit(self._features(X), y)
        return self

    def predict(self, amp):
        if self.branch == 'structured':
            return self.clf.predict(self._structured_features(np.asarray(amp, dtype=np.float64)))
        if self.branch == 'trca':
            return self.classes[self._trca_scores(self._views(np.asarray(amp, dtype=np.float64))).argmax(1)]
        return self.clf.predict(self._features(self._views(np.asarray(amp, dtype=np.float64))))


def _selfcheck():
    """Separable synthetic data: class 1 has extra power on channel 0 in Q-atom 0 (power branch) / a fixed signed
    offset on channel 1 (signed branch); both branches must classify held-out trials well above chance."""
    rng = np.random.default_rng(0)
    n, P, C, S = 80, 6, 4, 3
    y = np.repeat([0, 1], n // 2)
    amp = rng.standard_normal((n, P, C, S, 2))
    amp[y == 1, :, 0, 0, :] *= 3.0
    amp[y == 1, :, 1, :, 0] += 1.0
    tr, ev = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    for branch in ('power', 'signed', 'structured'):
        acc = (ClosedFormHead(branch, nfilter=2).fit(amp[tr], y[tr]).predict(amp[ev]) == y[ev]).mean()
        assert acc > 0.8, (branch, acc)
    # trca: 4 classes, each a different phase-locked time course (over patches) on a fixed channel mix, in noise
    y4 = np.arange(n) % 4
    t = np.arange(P)
    amp4 = rng.standard_normal((n, P, C, S, 2))
    mix = rng.standard_normal(C)
    for c in range(4):
        amp4[y4 == c, :, :, 0, 0] += 1.5 * np.sin(2 * np.pi * (c + 1) * t / P)[:, None] * mix
    tr4 = np.arange(n) < 60
    acc = (ClosedFormHead('trca', nfilter=1).fit(amp4[tr4], y4[tr4]).predict(amp4[~tr4]) == y4[~tr4]).mean()
    assert acc > 0.8, ('trca', acc)
    print('closed_form selfcheck ok')


if __name__ == '__main__':
    _selfcheck()
