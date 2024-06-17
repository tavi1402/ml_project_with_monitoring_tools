import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, recall_score

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Set our tracking server URI for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Loan Default Prediction")

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(n_jobs=6),
                "NaiveBayes Classifier": BernoulliNB(),
                "RandomForest Classifier": RandomForestClassifier(n_jobs=6, max_depth=17),
                "LGBM Classifier": LGBMClassifier(n_estimators=200, n_jobs=6, max_depth=15),
                "XGB Classifier": XGBClassifier(n_estimators=200, n_jobs=6, max_depth=15),
            }

            logging.info("Starting Model Training")
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Get best model based on recall score
            best_model_recall_score = max(sorted(model_report.values()))

            # Get best model's name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_recall_score)
            ]
            best_model = models[best_model_name]

            logging.info("Model Training completed")
            
            if best_model_recall_score < 0.5:
                logging.info("No best model found")
            else:
                # Start an MLflow run
                with mlflow.start_run():
                    # Log model parameters and metrics
                    mlflow.log_param("best_model", best_model_name)
                    mlflow.log_metric("best_model_recall_score", best_model_recall_score)

                    # Train and log the best model
                    best_model.fit(X_train, y_train)
                    y_pred = best_model.predict(X_test)
                    
                    # Infer signature for the logged model
                    signature = infer_signature(X_train, y_pred)
                    
                    mlflow.sklearn.log_model(best_model, "best_model", signature=signature)

                    acc = accuracy_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    
                    # Log accuracy and recall
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("recall", recall)

                    # Save the best model
                    save_object(
                        file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model
                    )

                    logging.info(f"Best model: {best_model_name}")
                    logging.info(f"Accuracy: {acc}")
                    logging.info(f"Recall: {recall}")
                    
                    return (acc, recall)

        except Exception as e:
            raise CustomException(e, sys)
