import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp


class RMR:
    """Running Mean Regression"""

    def __init__(self, x, y, h):
        self.x = np.array(x)
        self.y = np.array(y)
        self.h = h

    def running_mean(self, z):
        return 1 if abs(z) < 1 else 0

    def predict(self, X):
        kernels = np.array(
            [self.running_mean((np.linalg.norm(xi - X)) / self.h) for xi in self.x]
        )
        sum_kernels = np.sum(kernels)

        if sum_kernels > 0:
            weights = np.array(
                [len(self.x) * (kernel / sum_kernels) for kernel in kernels]
            )
        else:
            weights = np.zeros(len(kernels))

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
