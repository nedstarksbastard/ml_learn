from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_neighbors':[5,10,15],'weights':['uniform', 'distance']}]

kneighbor_cls = KNeighborsClassifier()

grid_search = GridSearchCV(kneighbor_cls, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_