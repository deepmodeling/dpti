From apache/airflow
USER root
# install postgresql
RUN apt-get update
RUN apt-get -y install wget vim
RUN sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
RUN apt-get update
RUN apt-get -y install postgresql-16
RUN apt-get -y install git

# create airflow postgresql database
RUN service postgresql start \
&& service postgresql status \
&& sudo -u postgres psql -U postgres --command="CREATE DATABASE airflow_db;" \
&& sudo -u postgres psql -U postgres --command="CREATE USER airflow WITH PASSWORD 'airflow_pw';" \
&& sudo -u postgres psql -U postgres --command="GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;" \
&& sudo -u postgres psql -U postgres --command="ALTER DATABASE airflow_db OWNER TO airflow;"

ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:airflow_pw@localhost:5432/airflow_db"


# download dpti source code and install
USER airflow
WORKDIR /home/airflow
RUN git clone https://mirror.ghproxy.com/github.com/deepmodeling/dpti.git

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN cd dpti && pip install .

USER root
RUN apt-get -y install procps

ENV AIRFLOW_HOME=/home/airflow/airflow/

RUN service postgresql start && service postgresql status \
&& sudo --preserve-env=AIRFLOW__DATABASE__SQL_ALCHEMY_CONN --login -u airflow airflow db init

# configure airflow
USER airflow
RUN sed -i "s|sqlite:////home/airflow/airflow/airflow.db|"$AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"|" /home/airflow/airflow/airflow.cfg
RUN sed -i "s|SequentialExecutor|LocalExecutor|" /home/airflow/airflow/airflow.cfg
RUN sed -i "s|load_examples = True|load_examples = False|" /home/airflow/airflow/airflow.cfg

# initialize airflow database
USER root
RUN service postgresql start && service postgresql status \
&& sudo --login -u airflow airflow db init

RUN service postgresql start && service postgresql status \
    && sudo --login -u airflow airflow users create \
    --username airflow \
    --password airflow \
    --firstname Peter \
    --lastname Parker \
    --role Admin \
    --email spiderman@superhero.org

# install dpti DAGs
USER airflow
RUN mkdir -p ~/airflow/dags/
RUN ln -s ~/dpti/workflow/DpFreeEnergy.py  ~/airflow/dags/
RUN ln -s ~/dpti/workflow/DpFreeEnergyWater.py  ~/airflow/dags/

# setup docker image
USER root
RUN apt-get -y install gosu
COPY entry.sh /entry.sh

ENTRYPOINT ["/entry.sh"]

# ENTRYPOINT service postgresql start && service postgresql status && exec gosu airflow "$@"

# ENTRYPOINT ["/bin/bash"]


