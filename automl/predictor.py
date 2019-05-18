import numpy as np
import pandas as pd
from sklearn import decomposition, preprocessing
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.metrics import roc_curve, auc, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
from automl.gs_params import params
import warnings
warnings.filterwarnings('ignore')

class prediction(object):
    def __init__(self, dataset, target, type_of_estimator=None):
        if (type_of_estimator == None):
            if(dataset[target].nunique() > 10):
                self.type = "continuous"
            else:
                self.type = "classifier"
        else:
            if (type_of_estimator == "continuous" or type_of_estimator =="classifier"):
                self.type = type_of_estimator
            else:
                print('Invalid value for "type_of_estimator". Please pass in either "continuous" or "classifier". You passed in: ' + type_of_estimator)
        self.dataset = dataset
        self.result = {}
        self.reducedDataset = None
        self.withoutOutliers = None
        self.clean()
        self.target = target
        self.grid_S = params()
        self.Outliers()
        self.Y = dataset[target].values
        self.X = dataset.loc[:, dataset.columns != target].values
        self.params = params()
        self.reduction()
        self.train(self.X,self.Y)

    def reduction(self):
        numberOfComponent = len(
            self.dataset.loc[:, self.dataset.columns != self.target].columns)
        total_variance_explained = 0
        X_temp = None
        dimension = 1
        std_scale = preprocessing.StandardScaler().fit(self.X)
        X_scaled = std_scale.transform(self.X)
        V = np.sum(np.var(X_scaled, axis=0))
        while(total_variance_explained < 90 and dimension < numberOfComponent):
            dimension = dimension + 1
            pca = decomposition.PCA(n_components=dimension)
            pca.fit(X_scaled)
            X_projected = pca.transform(X_scaled)
            explained_variance = np.var(X_projected, axis=0)
            total_variance_explained = np.sum(explained_variance)/V
            if (total_variance_explained > 90):
                X_temp = pca.transform(X_scaled)
        self.reducedDataset = X_temp

    def clean(self):
        number_of_Nan = self.dataset.isnull().sum().sum()
        pourcentage_of_Nan = (number_of_Nan/self.dataset.count().sum())*100
        print('NaNs represent ' + str(pourcentage_of_Nan) + ' pourcentage of dataset')
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

    def train(self, X, Y):
        models = {}
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.20, random_state=42)
        if (self.type == "continuous"):
            
            perf = self.modelLasso(X_train, y_train, X_test, y_test)
            models.update({'Lasso': perf})
            #perf = self.modelRandomForestR(X_train, y_train, X_test, y_test)
            #models.update({'RandomForestRegressor': perf})
            perf = self.modelElasticNet(X_train, y_train, X_test, y_test)
            models.update({'ElasticNet': perf})
            perf = self.modelLinearSVR(X_train, y_train, X_test, y_test)
            models.update({'LinearSVR': perf})
            perf = self.modelLinearRegression(X_train, y_train, X_test, y_test)
            models.update({'LinearRegression': perf})
            perf = self.modelAdaBoostRegressor(X_train, y_train, X_test, y_test)
            models.update({'AdaBoostRegressor': perf})
        elif (self.type == "classifier"):
            perf = self.modelLinearSVC(X_train, y_train, X_test, y_test)
            models.update({'SVC': perf})
            perf = self.modelRandomForestClassifier(X_train, y_train, X_test, y_test)
            models.update({'RandomForestClassifier': perf})
            perf = self.modelLogisticRegressor(
                X_train, y_train, X_test, y_test)
            models.update({'LogisticRegressor': perf})

        if (self.type == "classifier"):
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp:
                    temp = models[key]['accurracy']
                    final_model1 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp and models[key]!=final_model1:
                    temp = models[key]['accurracy']
                    final_model2 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy'] > temp and models[key]!=final_model1 and models[key]!=final_model2:
                    temp = models[key]['accurracy']
                    final_model3 = models[key]
            if (final_model1['name'] != 'RandomForestClassifier'):
                self.rocCurve(final_model1['model'],
                          X_train, y_train, X_test, y_test)
            if (final_model2['name'] != 'RandomForestClassifier'):
                self.rocCurve(final_model2['model'],
                          X_train, y_train, X_test, y_test)
            if (final_model3['name'] != 'RandomForestClassifier'):
                self.rocCurve(final_model3['model'],
                          X_train, y_train, X_test, y_test)
        if (self.type == "continuous"):
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp:
                    temp = models[key]['accurracy']['accurracy']
                    final_model1 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp and models[key]!=final_model1:
                    temp = models[key]['accurracy']['accurracy']
                    final_model2 = models[key]
            temp = 0
            for key in models:
                if models[key]['accurracy']['accurracy'] > temp and models[key]!=final_model1 and models[key]!=final_model2:
                    temp = models[key]['accurracy']['accurracy']
                    final_model3 = models[key]
        self.result = {'First' : final_model1 , 'Second': final_model2, 'Third' : final_model3}
        
        print('first model:')
        print(final_model1)
        print('second model:')
        print(final_model2)
        print('third model:')
        print(final_model3)
        print('use .result to access models')

    def evaluate(self, model, X_test, y_test):
        results = cross_val_score(
            model, X_test, y_test, cv=KFold(n_splits=10), n_jobs=1)
        result = np.mean(list(filter(lambda x: x > 0, results)))
        if (self.type=="continuous"):
            mse_test = mean_squared_error(y_test, model.predict(X_test))
            result = {'accurracy': result, 'rmse': np.sqrt(mse_test)}
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
        performance = {'model': bestmodel, 'accurracy': result , 'name': 'lasso'}
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
                return {'model': best_grid, 'accurracy': grid_accuracy,'name': 'RandomForestRegressor'}
            else:
                return {'model': random_grid, 'accurracy': random_accuracy,'name': 'RandomForestRegressor'}


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
        return {'model': best_grid, 'accurracy': grid_accuracy,'name': 'LogisticRegressor'}

    def rocCurve(self, model, X_train, y_train, X_test, y_test):
        y_train1 = label_binarize(
            y_train, list(range(0, self.dataset[self.target].nunique())))
        y_test1 = label_binarize(
            y_test, list(range(0, self.dataset[self.target].nunique())))
        n_classes = y_train1.shape[1]
        classifier = OneVsRestClassifier(model)
        y_score = classifier.fit(
            X_train, y_train1).decision_function(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if (n_classes == 1):
            fpr[0], tpr[0], _ = roc_curve(y_test1[:, 0], y_score)
            roc_auc[0] = auc(fpr[0], tpr[0])
        else:
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test1.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def Outliers(self):
        Q1 = self.dataset.quantile(0.25)
        Q3 = self.dataset.quantile(0.75)
        IQR = Q3 - Q1
        self.withoutOutliers = self.dataset[~((self.dataset < (
            Q1 - 1.5 * IQR)) | (self.dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
        pourcentage_of_outliers = (self.withoutOutliers.count()[
                                   self.withoutOutliers.columns[0]]/self.dataset.count()[self.dataset.columns[0]])*100
        print('there is ' + str(pourcentage_of_outliers) +
              ' pourcetage of rows with outliers')

    def modelRandomForestClassifier(self, X_train, y_train, X_test, y_test):
        lr = RandomForestClassifier()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['RandomForestClassifier'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name' : 'RandomForestClassifier'}

    def modelLinearRegression(self, X_train, y_train, X_test, y_test):
        lr = LinearRegression()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearRegression'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy, 'name':LinearRegression}

    def modelAdaBoostRegressor(self, X_train, y_train, X_test, y_test):
        lr = AdaBoostRegressor()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['AdaBoostRegressor'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'AdaBoostRegressor'}

    def modelElasticNet(self, X_train, y_train, X_test, y_test):
        lr = ElasticNet()
        grid = GridSearchCV(
            estimator=lr, param_grid= self.params['ElasticNet'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'ElasticNet'}

    def modelLinearSVR(self, X_train, y_train, X_test, y_test):
        lr = LinearSVR()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVR'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearSVR'}

    def modelLinearSVC(self, X_train, y_train, X_test, y_test):
        lr = LinearSVC()
        grid = GridSearchCV(
            estimator=lr, param_grid=self.params['LinearSVC'], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_grid = grid.best_estimator_
        grid_accuracy = self.evaluate(best_grid, X_test, y_test)
        return {'model': best_grid, 'accurracy': grid_accuracy,'name' :'LinearS'}
