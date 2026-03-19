import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

if __name__ == "__main__":
    # Start an MLflow experiment
    mlflow.set_experiment("My_First_Project")

    with mlflow.start_run():
        # 1. Define dummy data & parameters
        n_estimators = 100
        mlflow.log_param("n_estimators", n_estimators)

        # 2. "Train" a simple model
        data = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(data[["x"]], data["y"])

        # 3. Log a metric
        mse = mean_squared_error([2, 4, 6], model.predict(data[["x"]]))
        mlflow.log_metric("mse", mse)

        # 4. Save the model artifact
        mlflow.sklearn.log_model(model, "random-forest-model")

        print(f"Run completed! MSE: {mse}")
