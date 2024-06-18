# import numpy as np
# from textwrap import dedent
# # import pendulum
# import datetime
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from src.logger import logging
# from src.pipeline.training_pipeline import TrainingPipeline

# training_pipeline=TrainingPipeline()

# with DAG(
#     "Loan_Default_Prediction_Pipeline",   # Name
#     default_args={"retries": 2},
#     description="It is my training pipeline",   # Basic Description
#     schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
#     # start_date=pendulum.datetime(2024, 6, 16, tz="UTC"),
#     star_date=datetime.datetime(2024, 6, 16, tzinfo=datetime.timezone.utc),
#     catchup=False,
#     tags=["machine_learning ","classification","Loan_default"],
# ) as dag:
    
#     dag.doc_md = __doc__

#     # Define the functions

#     def data_ingestion(**kwargs):
#         ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
#         train_data_path,test_data_path=training_pipeline.start_data_ingestion() # Call the Data Ingestion Function
#         ti.xcom_push("data_ingestion_artifact", {"train_data_path":train_data_path,"test_data_path":test_data_path})    # Push arguments to next function
#         logging.info(f'Data Ingestion Sucessful')

#     def data_transformations(**kwargs):
#         ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
#         data_ingestion_artifact=ti.xcom_pull(task_ids="data_ingestion",key="data_ingestion_artifact")   # Pull the arguments from previous function
#         train_arr,test_arr=training_pipeline.start_data_transformation(data_ingestion_artifact["train_data_path"],data_ingestion_artifact["test_data_path"])    # Call the Data Transformation function
#         train_arr=train_arr.tolist()
#         test_arr=test_arr.tolist()
#         ti.xcom_push("data_transformations_artifcat", {"train_arr":train_arr,"test_arr":test_arr})  # Push arguments to next function
#         logging.info(f'Data Transformation Sucessful')

#     def model_trainer(**kwargs):
#         ti = kwargs["ti"]       # Dictionary (ti = Task Instance/Information)
#         data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")   # Pull the arguments from previous function
#         train_arr=np.array(data_transformation_artifact["train_arr"])
#         test_arr=np.array(data_transformation_artifact["test_arr"])
#         training_pipeline.start_model_training(train_arr,test_arr)  # Call the Model trainer function
#         logging.info(f'Model Training Sucessful')

#     # Define the Tasks
#     data_ingestion_task = PythonOperator(
#         task_id="data_ingestion",
#         python_callable=data_ingestion,
#     )
#     data_ingestion_task.doc_md = dedent(
#         """\
#     #### Ingestion task
#     this task creates a train and test file.
#     """
#     )

#     data_transform_task = PythonOperator(
#         task_id="data_transformation",
#         python_callable=data_transformations,
#     )
#     data_transform_task.doc_md = dedent(
#         """\
#     #### Transformation task
#     this task performs the transformation
#     """
#     )

#     model_trainer_task = PythonOperator(
#         task_id="model_trainer",
#         python_callable=model_trainer,
#     )
#     model_trainer_task.doc_md = dedent(
#         """\
#     #### model trainer task
#     this task perform training
#     """
#     )

# # Define the task dependencies
# data_ingestion_task >> data_transform_task >> model_trainer_task


from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow/src')
from src.pipeline.training_pipeline import TrainingPipeline
from src.exception import CustomException

pipeline = TrainingPipeline()

def start_data_ingestion():
    try:
        train_data_path, test_data_path = pipeline.start_data_ingestion()
        return train_data_path, test_data_path
    except Exception as e:
        raise CustomException(e, sys)
        
def start_data_transformation(ti):
    try:
        train_data_path, test_data_path = ti.xcom_pull(task_ids='start_data_ingestion')
        train_arr, test_arr = pipeline.start_data_transformation(train_data_path, test_data_path)
        return train_arr, test_arr
    except Exception as e:
        raise CustomException(e, sys)
        
def start_model_training(ti):
    try:
        train_arr, test_arr = ti.xcom_pull(task_ids='start_data_transformation')
        pipeline.start_model_training(train_arr, test_arr)
    except Exception as e:
        raise CustomException(e, sys)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='A machine learning training pipeline',
    schedule_interval=timedelta(weeks=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

start_data_ingestion_task = PythonOperator(
    task_id='start_data_ingestion',
    python_callable=start_data_ingestion,
    dag=dag,
)

start_data_transformation_task = PythonOperator(
    task_id='start_data_transformation',
    python_callable=start_data_transformation,
    provide_context=True,
    dag=dag,
)

start_model_training_task = PythonOperator(
    task_id='start_model_training',
    python_callable=start_model_training,
    provide_context=True,
    dag=dag,
)

start_data_ingestion_task >> start_data_transformation_task >> start_model_training_task
