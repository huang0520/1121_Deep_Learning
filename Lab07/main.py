# %% [markdown]
# # Lab 07: KNN, SVM, Data Preprocessing, and Scikit-learn Pipeline

# 112501533 黃思誠

# %%

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mushroom/agaricus-lepiota.data",
    header=None,
    engine="python",
)
column_name = [
    "classes",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]
df.columns = column_name
df.head()

# %% [markdown]
# ## Preprocessing

# 直接刪除有缺失值的資料

# %%
df = df.replace("?", np.nan)
df = df.dropna()

x = df.iloc[:, 1:].to_numpy()
y = df.iloc[:, 0].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# %% [markdown]
# ## Train and predict

# 使用 OneHotEncoder 將資料轉換成數值

# 分別使用 KNN 和 SVM 進行分類，可以發現 SVM 的準確率較高

# %%
categories = df.columns[1:].tolist()
categories_features_idx = [i for i in range(len(categories))]

ohe = ColumnTransformer(
    [
        (
            "ohe",
            OneHotEncoder(sparse_output=False),
            categories_features_idx,
        ),
    ],
    remainder="passthrough",
)
pipe_knn = Pipeline(
    [
        ("ohe", ohe),
        ("scl", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=10, p=2, metric="minkowski")),
    ]
)
pipe_svm = Pipeline(
    [
        ("ohe", ohe),
        ("scl", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=0, gamma=0.001, C=100.0)),
    ]
)

# %%
pipe_knn.fit(x_train, y_train)
y_pred = pipe_knn.predict(x_test)
print(f"KNN Misclassified samples: {(y_test != y_pred).sum()}")
print(f"KNN Test Accuracy: {accuracy_score(y_test, y_pred):.3f}", end="\n\n")

pipe_svm.fit(x_train, y_train)
y_pred = pipe_svm.predict(x_test)
print(f"SVM Misclassified samples: {(y_test != y_pred).sum()}")
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
