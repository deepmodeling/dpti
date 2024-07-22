
# useful command to use this docker file and image

## docker installation:
see: https://docs.docker.com/engine/install/


## pull docker image:

```docker pull deepmodeling/dpti```

note: maintained on https://hub.docker.com/r/deepmodeling/dpti



## start container and name it as 'dpti'
note: 9999:8000 means map 8080 port in the docker container to the 9999 port on the host machine

```docker run --name dpti -p 9999:8080 -it deepmodeling/dpti:latest /bin/bash```

## enter this container named 'dpti'

```docker exec -it dpti /bin/bash```

## useful files and dirs
airflow home dir: /root/airflow
latest source code dir: /root/dpti

## manually install dpti software locally

```cd /root/dpti && pip install .```

## command related to airflow
```
airflow webserver --hostname 0.0.0.0 --port 8080 &
airflow scheduler &
```

## for developer: build this image with Dockerfile: cd into dpti/docker/ dir

```docker build .  --tag deepmodeling/dpti:latest```
