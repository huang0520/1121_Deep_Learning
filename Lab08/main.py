# %% [markdown]
# # Lab08: Cross Validation & Ensembling

# 112501533 黃思誠

# %%

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# %%
# Load data
df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

x_train, y_train = (
    df_train.drop(columns=["Competitor"]).to_numpy(),
    df_train["Competitor"],
)
x_test, y_test = df_test.drop(columns=["Competitor"]).to_numpy(), df_test["Competitor"]

label_to_idx = {
    "Kate": 0,
    "Bob": 1,
    "Mark": 2,
    "Sue": 3,
}

y_train = y_train.map(label_to_idx).to_numpy()
y_test = y_test.map(label_to_idx).to_numpy()

# %% [markdown]
# ## Voting

# %%
pipe_dt = Pipeline([["clf", DecisionTreeClassifier(max_depth=3, random_state=0)]])
pipe_knn = Pipeline(
    [["sc", StandardScaler()], ["clf", KNeighborsClassifier(n_neighbors=5)]]
)

clf = VotingClassifier(
    estimators=[("dt", pipe_dt), ("knn", pipe_knn)], voting="soft", weights=[1, 3]
)
score = cross_val_score(
    estimator=clf, X=x_train, y=y_train, cv=10, n_jobs=-1, scoring="accuracy"
)
print(f"[Voting] Accuracy: {score.mean():.3f} +/- {score.std():.3f}")

# %% [markdown]
# ## Bagging

# %%
tree = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=0)
bag = BaggingClassifier(
    estimator=tree,
    n_estimators=500,
    max_samples=0.7,
    bootstrap=True,
    max_features=1.0,
    bootstrap_features=False,
    n_jobs=1,
    random_state=1,
)

score = cross_val_score(
    estimator=bag, X=x_train, y=y_train, cv=10, n_jobs=-1, scoring="accuracy"
)
print(f"[Bagging] Accuracy: {score.mean():.3f} +/- {score.std():.3f}")

# %% [markdown]
# ## Boosting

# %%
tree = DecisionTreeClassifier(criterion="entropy", max_depth=1)
ada = AdaBoostClassifier(estimator=tree, n_estimators=500)

score = cross_val_score(
    estimator=ada, X=x_train, y=y_train, cv=10, n_jobs=-1, scoring="accuracy"
)
print(f"[AdaBoost] Accuracy: {score.mean():.3f} +/- {score.std():.3f}")


# %% [markdown]
# ## Grid Search

# %%
tree_depth = [1, 3, 5, 7, 9]
tree = DecisionTreeClassifier(criterion="entropy")
ada = AdaBoostClassifier(estimator=tree, n_estimators=500)
gs = GridSearchCV(
    estimator=ada,
    param_grid={
        "estimator__max_depth": tree_depth,
    },
)

gs = gs.fit(x_train, y_train)

print("[GridSearch Test Score]")
for depth, score in zip(tree_depth, gs.cv_results_["mean_test_score"]):
    print(f"Depth: {depth} Score: {score:.3f}")
print()

clf = gs.best_estimator_
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(f"[Best GridSearch] Accuracy: {accuracy_score(y_test, y_pred):.3f}")
