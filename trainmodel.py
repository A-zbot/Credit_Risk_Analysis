import joblib
from sk.learn.linear_model import LogesticRegression
from sk.learn.random_ensemble import RandomForestClassifier
from sk.learn.tree import DecisionTreeClassifier
from config import RANDOM_STATE, MODEL_DIR

def build_models():
    models = {
        'logistic_regression': LogesticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        'decision_tree': DecisionTreeClassifier(random_state=RANDOM_STATE)
    }
    return models

def train_single_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_all_models(models, X_train, y_train):
    trained_models = {}
    print("Training models...")
    print("=" * 60)
    for name, model in models.items():
        print(f"Training {name}...")
        train_model = train_single_model(model, X_train, y_train)
        trained_models[name] = train_model
        print(f"{name} trained successfully.")

    print("=" * 60)
    return trained_models