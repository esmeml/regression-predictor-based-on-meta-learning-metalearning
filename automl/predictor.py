import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor


class prediction(object):
    def __init__(self, dataset, target, type_of_estimator=None):
        if (type_of_estimator == None):
            if(dataset[target].nunique() > 10):
                self.type = "continuous"
            else:
                self.type = "classifier"
        else:
            self.type = type_of_estimator
        self.dataset = dataset
        self.clean()
        self.target = target
        self.Y = dataset[target].values
        self.X = dataset.loc[:, dataset.columns != target].values

    def reduction(self):
        numberOfComponent = len(
            self.dataset.loc[:, self.dataset.columns != self.target].columns)
        total_variance_explained = 90
        X_temp = self.X
        std_scale = preprocessing.StandardScaler().fit(self.X)
        X_scaled = std_scale.transform(self.X)
        V = np.sum(np.var(X_scaled, axis=0))
        while(total_variance_explained > 88):
            numberOfComponent = numberOfComponent-1
            pca = decomposition.PCA(n_components=numberOfComponent)
            pca.fit(X_scaled)
            X_projected = pca.transform(X_scaled)
            explained_variance = np.var(X_projected, axis=0)
            total_variance_explained = np.sum(explained_variance)/V
            if (total_variance_explained > 88):
                X_temp = pca.transform(X_scaled)
        self.X = X_temp

    def clean(self):
        for column in self.dataset.columns.values:
            # Replace NaNs with the median or mode of the column depending on the column type
            try:
                self.dataset[column].fillna(
                    self.dataset[column].median(), inplace=True)
            except TypeError:
                most_frequent = self.dataset[column].mode()
                # If the mode can't be computed, use the nearest valid value
                if len(most_frequent) > 0:
                    self.dataset[column].fillna(
                        self.dataset[column].mode()[0], inplace=True)
                else:
                    self.dataset[column].fillna(method='bfill', inplace=True)
                    self.dataset[column].fillna(method='ffill', inplace=True)

    def train(self):
        models = {}
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.Y, test_size=0.20, random_state=42)
        if (self.type == "continuous"):
            perf = self.modelLasso(X_train, y_train, X_test, y_test)
            models.update({'Lasso': perf})
            perf = self.modelRandomForestR(X_train, y_train, X_test, y_test)
            models.update({'RandomForestRegressor': perf})
            print(models)
        elif (self.type == "classifier"):
            perf = self.modelSVR(X_train, y_train, X_test, y_test)
            models.update({'SVR': perf})
            perf = self.modelLogisticRegressor(
                X_train, y_train, X_test, y_test)
            models.update({'LogisticRegressor': perf})
        temp = 0
        for key in models:
            if models[key]['accurracy'] > temp:
                temp = models[key]['accurracy']
                final_model = models[key]
        print(final_model)

    def evaluate(self, model, X_test, y_test):
        results = cross_val_score(
            model, X_test, y_test, cv=KFold(n_splits=5), n_jobs=1)
        result = np.mean(list(filter(lambda x: x > 0, results)))
        return result

    def modelLasso(self, X_train, y_train, X_test, y_test):
        lasso = Lasso(random_state=0, max_iter=10000)
        alphas = np.logspace(-4, -0.5, 30)
        tuned_parameters = [{'alpha': alphas}]
        n_folds = 5
        clf = GridSearchCV(lasso, tuned_parameters,
                           cv=n_folds, refit=False, return_train_score=True)
        grid_result = clf.fit(X_train, y_train)
        best_params = grid_result.best_params_
        bestmodel = Lasso(random_state=0, max_iter=10000,
                          alpha=best_params['alpha'])
        bestmodel.fit(X_train, y_train)
        result = self.evaluate(bestmodel, X_test, y_test)
        performance = {'model': bestmodel, 'accurracy': result}
        return performance

    def modelRandomForestR(self, X_train, y_train, X_test, y_test):
        rf = RandomForestRegressor(random_state=42)
        # Number of trees in random forest
        n_estimators = [int(x)
                        for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        #    Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf_random = RandomizedSearchCV(
            estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
        base_model.fit(X_train, y_train)
        base_accuracy = self.evaluate(base_model, X_test, y_test)
        best_random = rf_random.best_estimator_
        random_accuracy = self.evaluate(best_random, X_test, y_test)
        best_random = rf_random.best_params_
        if (random_accuracy < base_accuracy):
            return {'model': base_model, 'accurracy': base_accuracy}
        else:
            max_depth = list(filter(lambda x: x > 0, [int(x) for x in np.linspace(
                best_random['max_depth'], best_random['max_depth'] + 40, num=4)]))
            min_samples_leaf = list(filter(lambda x: x > 0, [
                                    best_random['min_samples_leaf']-1, best_random['min_samples_leaf'], best_random['min_samples_leaf']+1]))

            min_samples_split = list(filter(lambda x: x > 0, [
                                     best_random['min_samples_split']-2, best_random['min_samples_split'], best_random['min_samples_split']+2]))

            n_estimators = list(filter(lambda x: x > 0, [int(x) for x in np.linspace(
                best_random['n_estimators']-200, best_random['max_depth'] + 200, num=5)]))

            param_grid = {
                'bootstrap': best_random['bootstrap'],
                'max_depth': max_depth,
                'max_features': [2, 3],
                'min_samples_leaf': min_samples_leaf,
                'min_samples_split': min_samples_split,
                'n_estimators': n_estimators
            }
            rf = RandomForestRegressor(random_state=42)
            # Instantiate the grid search model
            grid_search = GridSearchCV(
                estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, return_train_score=True)
            grid_search.fit(X_train, y_train)
            best_grid = grid_search.best_estimator_
            grid_accuracy = self.evaluate(best_grid, X_test, y_test)
            if (grid_accuracy > random_accuracy):
                return {'model': best_grid, 'accurracy': grid_accuracy}
            else:
                return {'model': random_grid, 'accurracy': random_accuracy}

    def modelSVR(self, X_train, y_train, X_test, y_test):
        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid={
                'C': [0.1, 1, 100, 1000],
                'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        grid_result = gsc.fit(X_train, y_train)
        best_params = grid_result.best_params_
        best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
                       coef0=0.1, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
        best_svr.fit(X_train, y_train)
        result = self.evaluate(best_svr, X_test, y_test)
        performance = {'model': best_svr, 'accurracy': result}
        return performance

    def modelLogisticRegressor(self, X_train, y_train, X_test, y_test):
        dual = [True, False]
        max_iter = [100, 110, 120, 130, 140]
        param_grid = dict(dual=dual, max_iter=max_iter)
        lr = LogisticRegression(penalty='l2')
        grid = GridSearchCV(
            estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy}
