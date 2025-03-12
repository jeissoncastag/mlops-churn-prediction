import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  
import os

def load_data():
    """Cargar datos de prueba."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script
    data_path = os.path.abspath(os.path.join(current_dir, "..", "data", "churn_data_test.csv"))  # Ruta al archivo de datos
    data = pd.read_csv(data_path)
    return data

def preprocess_data(data, label_encoders, scaler):
    """Preprocesar datos usando los mismos preprocesadores que en el entrenamiento."""
    # Aplicar Label Encoding a las variables categóricas
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        le = label_encoders.get(col)
        if le:
            data[col] = le.transform(data[col])

    # Aplicar Standard Scaler a las variables numéricas
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    data[numeric_columns] = scaler.transform(data[numeric_columns])

    return data

def evaluate_model(model, X_test, y_test):
    """Evaluar el modelo y calcular métricas."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, f1, precision, recall

def main():
    # Configurar el tracking URI de MLflow
    mlflow.set_experiment("Churn Prediction - Evaluation")  # Nombre del experimento

    # Cargar datos de prueba
    data = load_data()

    # Cargar preprocesadores
    label_encoders = joblib.load("label_encoders.pkl")  # Cargar LabelEncoders
    scaler = joblib.load("scaler.pkl")                # Cargar StandardScaler

    # Preprocesar datos
    X_test = preprocess_data(data.drop("Churn", axis=1), label_encoders, scaler)
    y_test = data["Churn"]

    # Cargar el modelo entrenado
    model = mlflow.sklearn.load_model("runs:/<run_id>/model")  # Reemplaza <run_id> con el ID de la ejecución

    # Evaluar el modelo
    accuracy, f1, precision, recall = evaluate_model(model, X_test, y_test)

    # Registrar métricas en MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

    # Imprimir métricas
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    main()