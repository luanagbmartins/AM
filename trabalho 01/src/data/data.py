import luigi

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


class ProcessData(luigi.Task):
    categorical_features = luigi.ListParameter(
        default=[
            "workclass",
            "education",
            "marital",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]
    )

    numerical_features = luigi.ListParameter(
        default=[
            "age",
            "fnlwgt",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_week",
        ]
    )

    def requires(self):
        return RawData()

    def output(self):
        return {
            "x_train": luigi.LocalTarget("data/processed/x_train.npy"),
            "y_train": luigi.LocalTarget("data/processed/y_train.npy"),
            "x_test": luigi.LocalTarget("data/processed/x_test.npy"),
            "y_test": luigi.LocalTarget("data/processed/y_test.npy"),
        }

    def run(self):
        print("-----> Cleaning data...")

        df_train = pd.read_csv(self.input()["train"].path)
        df_test = pd.read_csv(self.input()["test"].path)

        label = {"<=50K": 0, ">50K": 1}
        df_train.label = [label[item] for item in df_train.label]

        label_t = {"<=50K.": 0, ">50K.": 1}
        df_test.label = [label_t[item] for item in df_test.label]

        x_train, y_train = self.process_data(df_train)
        x_test, y_test = self.process_data(df_test)

        with open(self.output()["x_train"].path, "wb") as f:
            np.save(f, x_train)
        with open(self.output()["y_train"].path, "wb") as f:
            np.save(f, y_train)

        with open(self.output()["x_test"].path, "wb") as f:
            np.save(f, x_test)
        with open(self.output()["y_test"].path, "wb") as f:
            np.save(f, y_test)
        print("-----> Done !")

    def process_data(self, df):
        scaler = StandardScaler()
        le = preprocessing.LabelEncoder()

        df_scaled = scaler.fit_transform(
            df[list(self.numerical_features)].astype(np.float64)
        )
        X_1 = df[list(self.categorical_features)].apply(le.fit_transform)

        y = df["label"].astype(np.int32)
        X = np.c_[df_scaled, X_1].astype(np.float32)
        return X, y


class RawData(luigi.Task):

    train_url: str = luigi.Parameter(
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    test_url: str = luigi.Parameter(
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    )

    def output(self):
        return {
            "train": luigi.LocalTarget("data/raw/train.csv"),
            "test": luigi.LocalTarget("data/raw/test.csv"),
        }

    def run(self):
        print("-----> Reading data...")

        train_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        test_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        )

        columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_week",
            "native_country",
            "label",
        ]

        df_train = pd.read_csv(
            train_url, skipinitialspace=True, names=columns, index_col=False
        )
        df_test = pd.read_csv(
            test_url, skiprows=1, skipinitialspace=True, names=columns, index_col=False
        )

        df_train.to_csv(self.output()["train"].path, index=False)
        df_test.to_csv(self.output()["test"].path, index=False)
