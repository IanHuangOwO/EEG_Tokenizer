"""Closed-form finetune head (2026-10-04): fit without SGD on the frozen stamp code (a, b), for few-shot.

training_params.finetune.fit = "closed_form", closed_form = {"branch": ..., "nfilter": 4}:
  power  -- induced power (MI): per stamp, the channel covariance of its a and b over the trial's patches (OAS
            shrinkage) -> CSP log-variance (nfilter filters) -> all stamps concatenated -> shrinkage LDA.
            The current head's stamp_power is the same form (power of a spatially filtered code) with SGD filters.
  signed -- phase-locked (P300): signed a, b per channel (stamps x patches as the time axis) -> xDAWN covariances
            (nfilter per class, OAS) -> tangent space -> logistic regression, L2 strength by 3-fold CV.
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

BRANCHES = ('power', 'signed')


class ClosedFormHead:
    def __init__(self, branch, nfilter=4):
        if branch not in BRANCHES:
            raise ValueError(f"closed_form.branch must be one of {BRANCHES}, got {branch!r}")
        self.branch, self.nfilter = branch, int(nfilter)

    def _views(self, amp):
        """amp [n, N', C, S, 2] -> power: [n, S, C, 2N'] (a and b over patches, per stamp);
        signed: [n, C, S*2*N'] (the signed code of every stamp as one long time axis)."""
        n, P, C, S, _ = amp.shape
        if self.branch == 'power':
            return amp.transpose(0, 3, 2, 4, 1).reshape(n, S, C, 2 * P)
        return amp.transpose(0, 2, 3, 4, 1).reshape(n, C, S * 2 * P)

    def _features(self, X):
        if self.branch == 'power':
            return np.concatenate([t.transform(X[:, s]) for s, t in enumerate(self.blocks)], 1)
        return self.blocks[0].transform(X)

    def fit(self, amp, y):
        X = self._views(np.asarray(amp, dtype=np.float64))
        k = len(np.unique(y))
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
        return self.clf.predict(self._features(self._views(np.asarray(amp, dtype=np.float64))))


def _selfcheck():
    """Separable synthetic data: class 1 has extra power on channel 0 in stamp 0 (power branch) / a fixed signed
    offset on channel 1 (signed branch); both branches must classify held-out trials well above chance."""
    rng = np.random.default_rng(0)
    n, P, C, S = 80, 6, 4, 3
    y = np.repeat([0, 1], n // 2)
    amp = rng.standard_normal((n, P, C, S, 2))
    amp[y == 1, :, 0, 0, :] *= 3.0
    amp[y == 1, :, 1, :, 0] += 1.0
    tr, ev = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    for branch in BRANCHES:
        acc = (ClosedFormHead(branch, nfilter=2).fit(amp[tr], y[tr]).predict(amp[ev]) == y[ev]).mean()
        assert acc > 0.8, (branch, acc)
    print('closed_form selfcheck ok')


if __name__ == '__main__':
    _selfcheck()
