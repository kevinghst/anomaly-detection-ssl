import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.datasets.base import BaseDataset
from src.utils.models_utils import TASK_TYPE

CAT_NAMES = [
    "tcp.flags",
    "mqtt.conack.flags",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.protoname",
    # determined empirically. have cardinality < 20
    "mqtt.conack.val",  # 0, 5
    "mqtt.conflag.cleansess",  # 0, 1
    "mqtt.conflag.passwd",  # 0, 1
    "mqtt.conflag.uname",  # 0, 1
    "mqtt.dupflag",  # 0, 1
    "mqtt.kalive",  # 0, 1, 2, 3, 234, 60, 65535
    "mqtt.msgtype",  # 0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 14
    "mqtt.proto_len",  # 0, 4
    "mqtt.qos",  # 0, 1
    "mqtt.retain",  # 0, 1
    "mqtt.ver",  # 0, 4
]

DROP_COLS = [
    "mqtt.msg",
    # below only have 1 value
    "mqtt.conack.flags.reserved",
    "mqtt.conack.flags.sp",
    "mqtt.conflag.qos",
    "mqtt.conflag.reserved",
    "mqtt.conflag.retain",
    "mqtt.conflag.willflag",
    "mqtt.sub.qos",
    "mqtt.suback.qos",
    "mqtt.willmsg",
    "mqtt.willmsg_len",
    "mqtt.willtopic",
    "mqtt.willtopic_len",
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
        self.tmp_file_names = ["train70_reduced.csv"]
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

        data["target"] = le.fit_transform(data["target"])
        self.y = data["target"].to_numpy()
        data = data.drop("target", axis=1)

        for i, col in enumerate(data.columns):
            if col in CAT_NAMES:
                data[col] = data[col].astype(str)

                unique_values = set(data[col].unique())
                # print(f"Column: '{col}' | Unique categories: {unique_values}")

                data[col] = le.fit_transform(data[col])

                self.cat_features.append(i)
                self.cardinalities.append((i, len(unique_values)))
            else:
                self.num_features.append(i)

                unique_values = set(data[col].unique())
                # if len(unique_values) < 20:
                #     print(f"Column: '{col}' \n Set size: {len(unique_values)} \n Unique categories: {unique_values} \n")

        self.X = data.to_numpy()

        # log transform and standardize numerical features
        self.X[:, self.num_features] = np.log1p(self.X[:, self.num_features])
        scaler = StandardScaler()
        self.X[:, self.num_features] = scaler.fit_transform(
            self.X[:, self.num_features]
        )

        self.N, self.D = self.X.shape

        self.num_or_cat = {idx: (idx in self.num_features) for idx in range(self.D)}

        self.is_data_loader = True
