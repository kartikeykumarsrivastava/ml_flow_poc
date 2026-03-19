# My First MLflow Project: Experiment Tracking

This project demonstrates the core pillars of **MLOps** using **MLflow**. It tracks a machine learning experiment, logging hyperparameters, metrics, and model artifacts into a centralized tracking system.

## 🚀 Purpose
The goal of this project is to move away from "unmanaged" machine learning and implement:
* **Reproducibility:** Ensuring the model can be recreated with the same results.
* **Traceability:** Tracking which parameters (e.g., `n_estimators`) led to which performance metrics (e.g., `MSE`).
* **Governance:** Storing models in a structured format for future deployment.

## 🛠 Tech Stack
* **Language:** Python 3.x
* **ML Library:** Scikit-learn
* **MLOps Tool:** MLflow
* **Backend Store:** SQLite (Relational database for experiment metadata)
* **Artifact Store:** Local File System (`mlruns/`)

## 📁 Project Structure
```text
ml-flow/
├── main.py              # Main training script with MLflow logging
├── mlflow.db            # SQLite database for experiment tracking
├── mlruns/              # Directory for model artifacts (serialized models)
├── .gitignore           # Prevents logging/environment files from entering Git
└── README.md            # Project documentation

⚙️ Setup & Installation
Create a Virtual Environment:

Bash
python -m venv mlflow-env
source mlflow-env/bin/activate  # Mac/Linux
# or
mlflow-env\Scripts\activate     # Windows
Install Dependencies:

Bash
pip install mlflow scikit-learn pandas
Run the Experiment:
Execute the training script to log data to the database:

Bash
python main.py
📊 Viewing the Dashboard
Because this project uses a SQL backend for robustness, you must point the MLflow UI to the database file:

Bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
After running this, open your browser and navigate to:
http://127.0.0.1:5000

💡 Key Learnings
Tracking URI: Learned how to switch from local file-based tracking to a SQL-based backend.

Artifacts vs Metadata: Understood that metrics/params go to the .db file, while the trained model (.pkl) is stored in the mlruns folder.

Experiment Isolation: Created a named experiment My_First_Project to separate runs from the default workspace.