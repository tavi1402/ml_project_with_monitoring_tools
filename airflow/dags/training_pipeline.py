import numpy as np
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline

training_pipeline=TrainingPipeline()

with DAG(
    "Loan_Default_Prediction_Pipeline",   # Name
    default_args={"retries": 2},
    description="It is my training pipeline",   # Basic Description
    schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
    start_date=pendulum.datetime(2024, 6, 16, tz="UTC"),
    catchup=False,
    tags=["machine_learning ","classification","Loan_default"],
) as dag:
    
    dag.doc_md = __doc__

    # Define the functions

    def data_ingestion(**kwargs):
        ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
        train_data_path,test_data_path=training_pipeline.start_data_ingestion() # Call the Data Ingestion Function
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":train_data_path,"test_data_path":test_data_path})    # Push arguments to next function
        logging.info(f'Data Ingestion Sucessful')

    def data_transformations(**kwargs):
        ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
        data_ingestion_artifact=ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")   # Pull the arguments from previous function
        train_arr,test_arr=training_pipeline.start_data_transformation(data_ingestion_artifact["train_data_path"],data_ingestion_artifact["test_data_path"])    # Call the Data Transformation function
        train_arr=train_arr.tolist()
        test_arr=test_arr.tolist()
        ti.xcom_push("data_transformations_artifcat", {"train_arr":train_arr,"test_arr":test_arr})  # Push arguments to next function
        logging.info(f'Data Transformation Sucessful')

    def model_trainer(**kwargs):
        ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")   # Pull the arguments from previous function
        train_arr=np.array(data_transformation_artifact["train_arr"])
        test_arr=np.array(data_transformation_artifact["test_arr"])
        training_pipeline.start_model_training(train_arr,test_arr)  # Call the Model trainer function
        logging.info(f'Model Training Sucessful')

    # Define the Tasks
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    this task creates a train and test file.
    """
    )

    data_transform_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformations,
    )
    data_transform_task.doc_md = dedent(
        """\
    #### Transformation task
    this task performs the transformation
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### model trainer task
    this task perform training
    """
    )

# Define the task dependencies
data_ingestion_task >> data_transform_task >> model_trainer_task