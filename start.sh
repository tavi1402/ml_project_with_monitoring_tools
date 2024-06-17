#!bin/sh

# Upgrade pip
pip install --upgrade pip

# Initialize the Airflow database
airflow db migrate

# Initialize the Airflow database
#airflow db init

nohup airflow scheduler &
airflow webserver
