import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib
from mlflow.models.signature import infer_signature

def load_data():
    """Cargar y preprocesar datos."""
    # Obtener la ruta absoluta al archivo de datos
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio del script
    data_path = os.path.abspath(os.path.join(current_dir, "..", "data", "churn_data.csv"))  # Ruta al archivo de datos
    data = pd.read_csv(data_path)

    # Label Encoding para variables categóricas
    categorical_columns = data.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Guardar el encoder para uso futuro

    # Standard Scaler para variables numéricas
    numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    return data.drop("Churn", axis=1), data["Churn"], label_encoders, scaler

def train_model(X_train, y_train):
    """Entrenar modelo."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluar modelo."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1

def main():
    # Cargar y preprocesar datos
    X, y, label_encoders, scaler = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar MLflow (usando backend local)
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run():
        # Entrenar modelo
        model = train_model(X_train, y_train)

        # Evaluar modelo
        accuracy, f1 = evaluate_model(model, X_test, y_test)

        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Crear un ejemplo de entrada (usando una fila de los datos de entrenamiento)
        input_example = X_train.iloc[[0]]  # Primera fila de los datos de entrenamiento

        # Inferir la firma del modelo
        signature = infer_signature(X_train, model.predict(X_train))

        # Registrar el modelo con firma y ejemplo de entrada
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Guardar los preprocesadores en archivos
        joblib.dump(label_encoders, "label_encoders.pkl")  # Guardar los LabelEncoders
        joblib.dump(scaler, "scaler.pkl")                 # Guardar el StandardScaler

        # Registrar los preprocesadores como artefactos
        mlflow.log_artifact("label_encoders.pkl")  # Registrar los LabelEncoders
        mlflow.log_artifact("scaler.pkl")         # Registrar el StandardScaler

        # Eliminar los archivos temporales (opcional)
        os.remove("label_encoders.pkl")
        os.remove("scaler.pkl")

if __name__ == "__main__":
    main()