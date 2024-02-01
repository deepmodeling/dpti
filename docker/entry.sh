#!/bin/bash
service postgresql start
service postgresql status
echo ">>>>>additional command: $@>>>>>"
# exec "$@"
echo ">>>>>starting airflow scheduler>>>>>"
gosu airflow airflow scheduler &
echo ">>>>>starting airflow webserver on port 8080>>>>>"
gosu airflow airflow webserver --hostname 0.0.0.0 --port 8080 &
echo ">>>>>exec additional command>>>>>"
exec gosu airflow "$@"
# su - airflow
# whoami
# echo "$@"
# exec "$@"

