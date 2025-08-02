import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("/workspaces/codespaces-blank/E2E_skillfy_demo/data/Iris.csv")
X = df.drop("Species", axis=1)
y = df["Species"]

model = LogisticRegression(max_iter=500)
model.fit(X, y)

with open("/workspaces/codespaces-blank/E2E_skillfy_demo/model/model.pkl", "wb") as f:
    pickle.dump(model, f)
