# Models/ml_models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

class MLModels:
    def __init__(self, model_type='random_forest', params=None):
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**params) if params else RandomForestClassifier()
        elif model_type == 'svm':
            self.model = SVC(**params) if params else SVC()
        else:
            raise ValueError("Unsupported ML model type")

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)

if __name__ == '__main__':
    # Dummy data test
    X_dummy = np.random.rand(100, 66)
    y_dummy = np.random.randint(0, 2, 100)
    
    rf = MLModels(model_type='random_forest', params={'n_estimators': 100})
    rf.train(X_dummy, y_dummy)
    print("Random Forest Accuracy:", rf.evaluate(X_dummy, y_dummy))
    
    svm = MLModels(model_type='svm', params={'C': 1.0, 'kernel': 'rbf'})
    svm.train(X_dummy, y_dummy)
    print("SVM Accuracy:", svm.evaluate(X_dummy, y_dummy))
