import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp


class GKR:
    """Gaussian Kernel Regression"""

    def __init__(self, x, y, h):
        self.x = np.array(x)
        self.y = np.array(y)
        self.h = h

    def gaussian_kernel(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def predict(self, X):
        kernels = np.array(
            [self.gaussian_kernel((np.linalg.norm(xi - X)) / self.h) for xi in self.x]
        )
        weights = np.array(
            [len(self.x) * (kernel / np.sum(kernels)) for kernel in kernels]
        )
        return 1 if np.dot(weights.T, self.y) / len(self.x) >= 0.5 else 0

    def score(self, X, y, _n_jobs=mp.cpu_count()):
        predict = Parallel(n_jobs=_n_jobs)(delayed(self.predict)(x) for x in tqdm(X))

        data = {"y_Actual": y.values, "y_Predicted": np.array(predict)}
        df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])

        confusion_matrix = pd.crosstab(
            df["y_Actual"],
            df["y_Predicted"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )

        return confusion_matrix
