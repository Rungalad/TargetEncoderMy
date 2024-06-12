class TGE:

    def __init__(self, fluctuate=0.1):
        self.columns_params = dict()
        self.fluctuate = fluctuate

    def fit(self, data: pd.DataFrame, y: np.ndarray) -> None:
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
            output[col] = output[col].apply(lambda x: self.columns_params[col][x])
        return output

e = TGE(fluctuate=0.1)
cat_cols = [i for i in X.columns if X[i].dtypes == 'object']
cat_feats = e.fit_transform(X[cat_cols], y)
