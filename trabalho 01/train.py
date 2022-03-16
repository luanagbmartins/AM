import luigi
import json
import numpy as np

import plotly.figure_factory as ff

from src.data import ProcessData
from src.models import GKR, RMR


MODEL = dict(gaussian_kernel=GKR, running_mean=RMR)


class NonparametricRegression(luigi.Task):
    selected_model = luigi.ChoiceParameter(
        choices=["gaussian_kernel", "running_mean"], default="gaussian_kernel"
    )
    n_jobs = luigi.IntParameter(default=1)

    def __init__(self, *args, **kwargs):
        super(NonparametricRegression, self).__init__(*args, **kwargs)

    def requires(self):
        return ProcessData()

    def output(self):
        return {
            "metrics": luigi.LocalTarget(
                "results/{}_metrics.json".format(self.selected_model)
            ),
        }

    def run(self):
        print("-----> Reading data")
        x_train = np.load(open(self.input()["x_train"].path, "rb"))
        y_train = np.load(open(self.input()["y_train"].path, "rb"))

        x_test = np.load(open(self.input()["x_test"].path, "rb"))
        y_test = np.load(open(self.input()["y_test"].path, "rb"))

        cm = {
            "accuracy": [],
            "recall": [],
            "precision": [],
        }

        print("-----> Train model")
        H = [0.1, 0.5, 1, 5, 10]
        for h in H:
            model = MODEL[self.selected_model](x_train, y_train, h)
            confusion_matrix = model.score(x_test, y_test, self.n_jobs)

            cm["accuracy"].append(
                (confusion_matrix[0][0] + confusion_matrix[1][1])
                / (
                    confusion_matrix[0][0]
                    + confusion_matrix[1][1]
                    + confusion_matrix[0][1]
                    + confusion_matrix[1][0]
                )
            )
            cm["recall"].append(
                confusion_matrix[0][0]
                / (confusion_matrix[0][0] + confusion_matrix[1][0])
            )
            cm["precision"].append(
                confusion_matrix[0][0]
                / (confusion_matrix[0][0] + confusion_matrix[0][1])
            )

            names = ["Falso", "Verdadeiro"]
            fig = ff.create_annotated_heatmap(confusion_matrix.values, x=names, y=names)
            fig.update_layout(
                yaxis=dict(categoryorder="category descending"),
                title="Matriz de Confusão",
            )
            fig.update_yaxes(title="Valor Real")
            fig.update_xaxes(title="Valor Predito")
            fig.show()
            fig.write_image(
                "images/{}_confusion_matrix_h-{}.svg".format(self.selected_model, h)
            )
            fig.write_image(
                "images/{}_confusion_matrix_h-{}.png".format(self.selected_model, h)
            )

        with open(self.output()["metrics"].path, "w") as outfile:
            json.dump(cm, outfile)
