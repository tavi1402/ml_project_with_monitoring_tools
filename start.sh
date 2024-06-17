#!/bin/sh

# Upgrade pip
pip install --upgrade pip

# Initialize the Airflow database
airflow db migrate

# Start the Airflow scheduler in the background
nohup airflow scheduler &

# Start the Airflow webserver
exec airflow webserver
