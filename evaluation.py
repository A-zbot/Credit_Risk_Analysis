import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from trainmodel import save_model
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    roc_auc = roc_auc_score(y_test, prediction)
    metrics = (
        "Accuracy" : accuracy,
        "Precision" : precision,
        "Recall" : recall,
        "F1" : f1,
        "Roc Auc" : roc_auc
    )
    return predictions, metrics

def confusion_matrix(y_test, prediction, model_name):
    matrix = (y_test, predictions)
    plt.figure(fix.size[6, 5]) 
    sns.heatmap(
        matrix, 
        annot = True
        fmt = "d",
        cmap = "Blues"
        )
    plt.title(f"{model_name} confusion_matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc(model, x_test, y_test, model_name):
    RocCurveDisplay(model, x_test, y_test)
    plt.title(f"{model_name} Roc Curve")    
    plt.titght_layout()
    plt.show()

def evaluate_all_models(trained_models, x_test, y_test):
    best_model = None
    best_score = 0
    print("\nMOodel Evaluation")
    print("=" * 60)
    for name, model in trained_models.items():
        print(f"\n{model_name}")
        metrics, predictions = evaluate_single_model(
            model,
            x_test,
            y_test
        )
        for metric, value in metrics.items():
            print(f"(metric:<12):{value:.4f}")
        print("\nClassification Report")
        print(classification_report(
            y_test, predictions
            )
        )
        plot_confusion_matrix(
            y_test,
            predictions,
            name
        )
        plot_roc_curve(
            model,
            x_test,
            y_test,
            name
        )
        if metrics["roc auc"] > best_score:
            best_score = metrics["roc auc"]
            best_model = model
        print("=" * 60)
        print(f"\nBest Model roc-auc: {best_score:.4f}")
        save_model(best_model)

    return best_model    



    

