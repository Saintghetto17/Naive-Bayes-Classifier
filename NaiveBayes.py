import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.proj3d import transform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from unicodedata import numeric


class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Implementation of Naive Bayes classifier. It can be useful for
    any classification problem with not too much features.
    This kind of classifier perfectly demonstrates the probability-method in ML.

    NOTE: It is not necessary to transform your continuous data to Standard distribution before fitting.
    """

    def __init__(self, transform=True):
        """
        Creates a naive bayes classifier.
        :param transform: Whether to transform the continuous data to Standard distribution before fitting.
        """
        super().__init__()
        self._classes = None
        self.conditioned_categorical_distribution = {}
        self.columns = []
        self.unique_features = {}
        self.conditioned_numerical_distribution = {}
        self.class_index = {}
        self.numeric = {}
        self.transform = transform

    def fit(self, X, y):
        """
        Fit model to training data. This process called training
        :param X: data of object-features pattern (num_elements, features)
        :param y: labels of each element (num_elements,)
        :return: self
        """
        X_copy = X.copy()
        self.numeric = set(X.select_dtypes(include=np.number).columns)
        if self.transform:
            self._scale_numeric(X_copy)

        self._classes = np.unique(y)
        self.class_index = dict(zip(self._classes, range(len(self._classes))))

        X_df, y_df = self._transform(X_copy, y)
        self.columns = X_df.columns

        y_df = y_df.flatten()

        for col in self.numeric:
            conditioned = X_df.groupby(y_df)[col]
            means = conditioned.mean()
            vars = conditioned.var()
            features_real = X[col].unique()
            for i, feature in enumerate(X_df[col].unique()):
                feature_real = features_real[i]
                prob_gauss = {}
                for cls in self._classes:
                    mean_conditioned = means[cls]
                    var_conditioned = vars[cls]
                    prob = 1 / (np.sqrt(2 * np.pi * var_conditioned)) * np.exp(
                        -(feature - mean_conditioned) ** 2 / (2 * var_conditioned))
                    prob_gauss[cls] = prob
                self.conditioned_numerical_distribution[(feature_real, col)] = prob_gauss

        for col in X_df.columns.difference(self.numeric):
            conditioned_categorical = X_df.groupby([y_df, col]).size()
            total_count = X_df.groupby(y_df).size()

            for feature in X_df[col].unique():

                prob_conditioned = {}

                for cls in self._classes:
                    count = conditioned_categorical.get((cls, feature), 0)

                    total_cls_count = total_count.get(cls, 1)
                    prob_conditioned[cls] = count / total_cls_count
                self.conditioned_categorical_distribution[(feature, col)] = prob_conditioned

        return self

    def predict(self, X):
        """
        Predicts label for each object
        :param X: data of object-features pattern (num_elements, features)
        :return: predicted labels
        """
        return self._classes[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """
        Predicts probability for each object of each label.
        :param X: data of object-features pattern (num_elements, features)
        :return: probability for each label for each object (num_elements, num_labels)
        """
        X_df = pd.DataFrame(X)
        num_samples = X_df.shape[0]
        num_classes = len(self._classes)
        prob_classes = np.zeros((num_samples, num_classes))

        for col in X_df.columns:
            feature_values = X_df[col].values
            distribution = self.conditioned_categorical_distribution if col not in self.numeric \
                else self.conditioned_numerical_distribution
            prob_values = np.zeros((num_samples, num_classes))
            for feature in np.unique(feature_values):
                if (feature, col) in distribution.keys():
                    prob_num = distribution[(feature, col)]
                    prob_num_array = np.array([prob_num.get(cls, 1e-10) for cls in self._classes])
                    prob_values[feature_values == feature] = prob_num_array
            prob_classes += np.log(prob_values + 1e-10)

        max_prob = np.max(prob_classes, axis=1, keepdims=True)
        exp_prob = np.exp(prob_classes - max_prob)
        normalized_prob = exp_prob / np.sum(exp_prob, axis=1, keepdims=True)

        return normalized_prob

    def _scale_numeric(self, X):
        scaler = StandardScaler()
        X[list(self.numeric)] = scaler.fit_transform(X[list(self.numeric)])

    def _transform(self, X, y):
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        X_df = X_df.reset_index()
        y_df = y_df.reset_index()
        X_df = X_df.drop(['index'], axis=1)
        y_df = y_df.drop(['index'], axis=1)
        return X_df, np.array(y_df)
