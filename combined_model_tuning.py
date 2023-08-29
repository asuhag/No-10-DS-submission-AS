import argparse
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def parse_list(string_values):
    try:
        return [int(i) for i in string_values.split(',')]
    except ValueError:
        return string_values.split(',')

# Load datasets
def load_data(path):
    return np.load(path)

# Set up argparse to accept command-line arguments
parser = argparse.ArgumentParser(description='Hyperparameter tuning for XGBoost Classifier.')
parser.add_argument('--X_train', type=str, required=True, help='Path to X_train data.')
parser.add_argument('--X_test', type=str, required=True, help='Path to X_test data.')
parser.add_argument('--y_train', type=str, required=True, help='Path to y_train data.')
parser.add_argument('--y_test', type=str, required=True, help='Path to y_test data.')
parser.add_argument('--max_depth', type=parse_list, default='3', help='Comma-separated list for max_depth values to try.')
parser.add_argument('--n_estimators', type=parse_list, default='100', help='Comma-separated list for n_estimators to try.')
parser.add_argument('--tree_method', type=parse_list, default='auto', help='Comma-separated list for tree_method to try.')
parser.add_argument('--predictor', type=parse_list, default='auto', help='Comma-separated list for predictor to try.')
parser.add_argument('--grow_policy', type=parse_list, default='depthwise', help='Comma-separated list for grow_policy to try.')

# Parse the arguments
args = parser.parse_args()

# Load datasets from the paths provided in the arguments
X_train = load_data(args.X_train)
X_test = load_data(args.X_test)
y_train = load_data(args.y_train)
y_test = load_data(args.y_test)

params = {
    'max_depth': args.max_depth,
    'n_estimators': args.n_estimators,
    'tree_method': args.tree_method,
    'predictor': args.predictor,
    'grow_policy': args.grow_policy
}

xgb_classifier = xgb.XGBClassifier()

grid_search = GridSearchCV(
    xgb_classifier, 
    param_grid=params, 
    cv=3,  # or other number of folds for cross validation
    verbose=2, 
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Results
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

best_classifier = grid_search.best_estimator_
prediction_xgb_prob = best_classifier.predict_proba(X_test)
