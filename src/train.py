import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("data/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression()
model.fit(X, y)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
