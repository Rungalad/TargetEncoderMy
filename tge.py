import numpy as np
import pandas as pd

class TGE:

    def __init__(self, fluctuate=0.1):
        self.columns_params = dict()
        self.fluctuate = fluctuate

    def fit(self, data_: pd.DataFrame, y: np.ndarray) -> None:
        data = copy.deepcopy(data_)
        data['target'] = y
        di = {}
        for col in data.columns:
            if col == 'target':
                continue
            print("remade ", col)
            prom = data[[col, 'target']]
            # prom = prom.groupby([col], as_index=False).agg({'target': ['mean', 'std']})
            prom = prom.groupby([col], as_index=False).mean()
            random = np.random.choice(np.linspace(-self.fluctuate, self.fluctuate, 1000))
            prom["target"] = prom["target"].apply(lambda x: x + random*x)
            subdi = dict(zip(prom[col].values, prom["target"].values))
            self.columns_params.update({col: subdi})

    def fit_transform(self, data: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        self.fit(data, y)
        output = copy.deepcopy(data)
        for col in self.columns_params:
            output[col] = output[col].apply(lambda x: self.columns_params[col][x])
        return output

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        output = copy.deepcopy(data)
        for col in self.columns_params:
            values = output[col].values
            remade_values = []
            mean_val = np.mean(list(self.columns_params[col].values()))
            for elem in values:
                if elem in self.columns_params[col]:
                    remade_values.append(self.columns_params[col][elem])
                else:
                    remade_values.append(mean_val)
            output[col] = remade_values
        return output

e = TGE(fluctuate=0.1)
cat_cols = [i for i in X_train.columns if X_train[i].dtypes == 'object']
cat_feats = e.fit_transform(X_train[cat_cols], y_train)
