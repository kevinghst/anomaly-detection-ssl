import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE

CAT_NAMES = [
    "tcp.flags",
    "mqtt.conack.flags",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.protoname",
]

DROP_COLS = [
    "mqtt.msg",
]


class MQTT(BaseDataset):
    """
    https://www.kaggle.com/api/v1/datasets/download/cnrieiit/mqttset

    MQTT Dataset
    Number of instances 8.5M
    Number of features 32
    """

    def __init__(self, args):
        super(MQTT, self).__init__(args)

        self.is_data_loaded = False
        self.tmp_file_names = ["train70.csv"]
        self.name = "mqtt"
        self.args = args
        self.task_type = TASK_TYPE.BINARY_CLASS

        self.cardinalities = []
        self.num_features = []
        self.cat_features = []

    def load(self):

        path = os.path.join(self.args.data_path, self.tmp_file_names[0])
        data = pd.read_csv(path)

        data = data.drop(columns=DROP_COLS)

        # collapse attack targets to 'illegitimate'
        data["target"] = np.where(
            data["target"] == "legitimate", "legitimate", "illegitimate"
        )

        le = LabelEncoder()

        for i, col in enumerate(data.columns):
            if col in CAT_NAMES:
                data[col] = data[col].astype(str)

                unique_values = set(data[col].unique())
                print(f"Column: '{col}' | Unique categories: {unique_values}")

                data[col] = le.fit_transform(data[col])

                self.cat_features.append(i)
                self.cardinalities.append((i, len(unique_values)))
            else:
                self.num_features.append(i)

        data["target"] = le.fit_transform(data["target"])
        self.y = data["target"].to_numpy()

        data = data.drop("target", axis=1)

        self.X = data.to_numpy()

        self.N, self.D = self.X.shape

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loader = True
