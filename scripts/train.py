import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
import tempfile
from mlflow.models.signature import infer_signature


def load_data():
    """Carga y preprocesa los datos."""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script
    data_path = os.path.abspath(os.path.join(current_dir, "..", "data", "churn_data.csv"))
    data = pd.read_csv(data_path)

    # Manejo de valores nulos
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
        data["TotalCharges"].fillna(data["TotalCharges"].median(), inplace=True)

    # Label Encoding para variables categóricas
    categorical_columns = data.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Standard Scaler para variables numéricas
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data.drop("Churn", axis=1), data["Churn"], label_encoders, scaler


def train_model(X_train, y_train):
    """Entrena un modelo RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo con métricas de accuracy y f1-score."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1


def main():
    # Cargar y preprocesar datos
    X, y, label_encoders, scaler = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Asegurar que las características sean tipo float64
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    # Configurar MLflow y ejecutar experimento
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")  # Verificar que el experimento se registre

        # Entrenar modelo
        model = train_model(X_train, y_train)

        # Evaluar modelo
        accuracy, f1 = evaluate_model(model, X_test, y_test)

        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Crear un ejemplo de entrada
        input_example = X_train.iloc[[0]]

        # Inferir la firma del modelo
        signature = infer_signature(X_train, model.predict(X_train))

        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Guardar y registrar los preprocesadores
        with tempfile.TemporaryDirectory() as temp_dir:
            encoders_path = os.path.join(temp_dir, "label_encoders.pkl")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")

            joblib.dump(label_encoders, encoders_path)
            joblib.dump(scaler, scaler_path)

            mlflow.log_artifact(encoders_path)
            mlflow.log_artifact(scaler_path)

if __name__ == "__main__":
    main()
