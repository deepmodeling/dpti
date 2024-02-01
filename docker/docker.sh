# useful docker scripts
# build image: run in dpti/docker/ dir
docker build .  --tag deepmodeling/dpti:latest

# start container called dpti
docker run --name dpti -p 9999:8080 -it deepmodeling/dpti:latest /bin/bash

# enter this container 
docker exec -it dpti /bin/bash
