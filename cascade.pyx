import numpy as np
from sklearn.ensemble.weight_boosting import _samme_proba
cimport numpy as np
cimport cython

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cascade(np.ndarray[DTYPE_t, ndim=2] X, clf):
    """
    :param X:
    :param clf:
    :return:
    An attempt at doing cascade classification in cython, it doesn't seem to be any faster than the
    native sklearn .predict() implementation
    """
    cdef:
        float p_t
        np.ndarray[DTYPE_t, ndim=2] z2 = np.zeros((1, 2))
        int N = X.shape[0]
        np.ndarray[DTYPE_t, ndim=1] out = np.zeros(N)
        int t, i
    for i in range(N):
        pred = z2
        for t, estimator in enumerate(clf.estimators_):

            x = X[i, :].reshape(1, -1)
            pred += _samme_proba(estimator, 2, x)
            p_t = pred[0, 1] / (t + 1)
            if p_t < -.2 and t > 8:
                print(t)

                break

        if p_t < 0:
            out[i] = 0
        else:
            out[i] = 1
    return out
