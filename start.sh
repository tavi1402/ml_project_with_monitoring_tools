#!bin/sh

# Initialize the Airflow database
airflow db init

nohup airflow scheduler &
airflow webserver
