import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Load data
titanic = pd.read_csv("train.csv")

# Train-test split
train_set, test_set = train_test_split(titanic, test_size=0.2, random_state=42)

# Separate label and features
titanic_label = train_set["Survived"].copy()
titanic_features = train_set.drop(columns=["Survived"], axis=1)

# Drop non-numerical columns to get numeric features only
titanic_num_features = train_set.drop(columns=["Name", "Sex", "Ticket", "Cabin", "Embarked", "Survived"])

# Custom transformer to add combined features
class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sibsp_index =3
        self.parch_index =4
        self.pclass_index =1 
        self.age_index =2

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        sibsp_parch = X[:, self.sibsp_index] + X[:, self.parch_index] + 1
        class_age = X[:, self.pclass_index] * X[:, self.age_index]
        return np.c_[X, sibsp_parch, class_age]

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# Define column lists
num_feature = list(titanic_num_features.columns)
cat_feature = [ "Sex","Embarked"]

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_feature),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feature),
])

# Transform training data
titanic_prepared = full_pipeline.fit_transform(titanic_features)

# Models
sgd_clf = SGDClassifier(loss="log_loss", random_state=42)
log_reg = LogisticRegression(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)

# Hyperparameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],         
#     'max_depth': [None, 5, 10, 20],          
#     'min_samples_split': [2, 5, 10],         
#     'min_samples_leaf': [1, 2, 4],           
#     'max_features': ['sqrt', 'log2', None],  
#     'bootstrap': [True, False]               
# }

param_grid = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Grid search/randomized search
random_search = RandomizedSearchCV(
    forest_clf,
    param_distributions = param_grid,
    n_iter=20,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

# grid_search = GridSearchCV(forest_clf, param_grid, cv=3, scoring='accuracy', return_train_score=True, n_jobs=-1)
random_search.fit(titanic_prepared, titanic_label)

print(random_search.best_estimator_)
print(random_search.best_score_)

# Best model from grid search
refined_model = random_search.best_estimator_

# # Prepare test data
titanic_test_label = test_set["Survived"].copy()
titanic_test_features = test_set.drop(columns=["Survived"], axis=1)
titanic_test_prepared = full_pipeline.transform(titanic_test_features)

# # # Predict on test set
titanic_test_predict = refined_model.predict(titanic_test_prepared)

# # # Evaluation
print("Test Accuracy:", accuracy_score(titanic_test_label, titanic_test_predict))
print("Classification Report:\n", classification_report(titanic_test_label, titanic_test_predict))

# # # Model predictions now 
titanic_test = pd.read_csv("test.csv")
titanic_test_features = full_pipeline.transform(titanic_test)
titanic_predicted = refined_model.predict(titanic_test_features)

submission = pd.DataFrame({
    "PassengerId" : titanic_test["PassengerId"],
    "Survived" : titanic_predicted
})

submission.to_csv("submission_2.csv", index=False)