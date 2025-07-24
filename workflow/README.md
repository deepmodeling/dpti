

# Airflow installation document. (deprecated)

airflow installation document. (deprecated)




## manually

```bash
 # copy dpti'workflow file
 cp /path-to-dpti/workflow/DpFreeEnergy.py ~/airflow/dags/

 # create a workdir and copy example files
 cp /path-to-dpti/examples/*json /path-to-a-work-dir/

 # start our airflow job
 cd /path-to-a-work-dir/
 cat ./airflow.sh

 airflow dags trigger  TI_taskflow  --conf $(printf "%s" $(cat FreeEnergy.json))

```


<a name="262831afc14feddc64db20cb6be8fd0d"></a>
##
<a name="ad87a3d8509a6920e3e849cb1b423f31"></a>
## 🕹install postgresql database
Airflow use relation database as  backend. And PostgreSQL is widely used in airflow community.<br />

<a name="nppPR"></a>
### install database
airflow's introduction on how to set up database backend: [apache-airflow:set up database](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html)
```bash
# install apache-airflow postgresql module
pip install apache-airflow-providers-postgres

# install postgresql
yum install postgresql

# enable postgresql service
systemctl start postgresql

# enter posgresql
psql

```


<a name="iFkxm"></a>
###  create database and database user
```sql
CREATE DATABASE airflow_db1;
CREATE USER airflow_user1 WITH PASSWORD 'airflow_user1';
GRANT ALL PRIVILEGES ON DATABASE airflow_db1 TO airflow_user1;
```
<a name="Glyxy"></a>
###
<a name="gQvK2"></a>
### configure airflow configure file to connect database
configure  ~/airflow/airflow.cfg<br />

```
# change the following item with the sql above
# sql_alchemy_conn = sqlite:////home/fengbo/airflow/airflow.db
# sql_alchemy_conn = postgres://airflow:airflow@localhost:5432/airflow
sql_alchemy_conn = postgresql+psycopg2://<user>:<password>@<host>:<port>/<db_name>
```
<a name="Yh8QG"></a>
### configure apache-airflow
reset db and webserver scheduler
```
# reset db
airflow db init
# -D flag represent daemonize
airflow webserver # -p 8080 -D
airflow scheduler # -D
```
<a name="QsNiC"></a>
### airflow webserver

If things work well, we could type the ip and prot in the web Browser and use the web service to monitor and manage the tasks operated by apache-airflow.

We refer to this doc [apache-airflow webserver guide](https://airflow.apache.org/docs/apache-airflow/stable/security/webserver.html) for further information.

after login(usually with default username and password:airflow, airflow)

[![template_webserver.png](https://s11.ax1x.com/2024/02/19/pFYmlWj.png)](https://imgse.com/i/pFYmlWj)

<br />

<a name="253840892aedd2058f97c95ac6ef6366"></a>

### ssh to use the webserver

Sometime, apache-airflow runs on remote machine and user can `ssh` to contect to the cloud server by command like ` ssh -L localhost:8080:localhost:8080 user1@67.xxx.xxx.25`  and visit [http://localhost:8080/](http://localhost:8080/) to monitor the free energy calculation tasks process.

### apache-airflow: further instruction

The backend of this software is based on the software `airflow`. The following command can start the calculation.

<br />
<br />The first command is used for calculate the free energy of solid.<br />The second command is used for calculate the free energy of liquid.<br />

```bash
airflow trigger_dag HTI_taskflow --conf $(printf "%s" $(cat FreeEnergy.json))
airflow trigger_dag TI_taskflow --conf $(printf "%s" $(cat FreeEnergy.meam.json))
```
#### FreeEnergy.json

We usually want to calculate the free energy of a metal at a specific pressure or temperature.  And the crystal structure of the metal can be various. For example,  we want to calculate the free energy of metal Sn of bcc structure at 200 K and 50000 bar (5GPa). In order to caculate the per atom free energy of metal Sn. First, We must prepare a configuration file named bcc.lmp and modify the [FreeEnergy.json](#ULX0o) or [FreeEnergyLiquid.json](#WuLBQ) and modify the key-value pair  like "structure": "bcc", "target_temp": 200, "target_press" : 50000.  And decide  whether to integrate along the  t(temperature) path  or along the p(pressure) path . Modify the "path" key-value pair for this.  The key-value pair "ensemble" for lammps MD simulation. Usually the ensemble shoule be consistent with the crystal intrinsic structure. That means we should set "npt-iso" for structure "bcc" to keep the simulation box changes simultaneously in x, y, z directions.

#### ti-path json
Modify the ti.t.json or ti.p.json, and change the key-value pair "temps" or "press" .  For ti.t.json, the tar_temp of FreeEnergy.json must be in the list  which the key-value pair "temps" of ti.t.json represents. And similarly for ti.p.json, the tar_press of FreeEnergy.json must be in the list which the key-value pair "temps" of ti.t.json represents.

#### workflow
1. Use the command `airflow trigger_dag`  mentioned above. This command will start a [airflow dag](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html).This dag is wrote and maintained by the dpti software developer. It is used to make the calculation to be done more autocally . The user could monitor the task state and calculation procedure at [a website](#2aabcbd6). The user can also rerun, restart, delete the whole calculation or some part of the calculations.
2. Wait until the calculation finish. Usually the whole procedure continues for about 6 to 10 hours. The calculations will be done autocally.
3. Find the results in [Results Show](#2aabcbd6) part. The user could use the tables and data of it and plot the curve.


## For airflow workflow:

Sometimes, we need to do **high-throughput** calculations(which means we need to calculate a series of temperature, pressure points for multiple phases).

It would be a great burden for users to execute these tasks manually and monitor the tasks' execution.

We provide the workflow tools based on apache-airflow workflow framework.

>we refer this docs [airflow official docs](https://airflow.apache.org/docs/apache-airflow/stable/index.html) for more instructions.


### TI_Workflow
We implement a workflow de
implemented at `workflow/DpFreeEnergy.py`

example dir and json:
```
cd examples/
cat examples/FreeEnergy.json
```

Requirement: setup apache-airflow or use the docker version dpti
```
docker run --name dpti -p 9999:8080 -it deepmodeling/dpti:latest /bin/bash
docker exec -it dpti /bin/bash
```
Then we provide a basic example for apache-airflow usage

```
# pwd at /home/airflow/
cd dpti/examples/
airflow dags trigger  TI_taskflow  --conf $(printf "%s" $(cat FreeEnergy.json))
```



**Input:** Lammps structure file.

**Output:** free energy values at given temperature and pressure.

**Parameters:** Given temperature(or the range), Given pressure(or therange), force field,Lammps simulation ensemble and etc.

we implement a workflow called TI_taskflow:
It includes these steps:
1. npt simulation to get lattice constant.
2. nvt simulation.
3. HTI: free energy at given temperature and pressure
4. TI: free energy values at the given range of temperature/pressure.

### website to manage and monitor these jobs.



## For airflow workflow:

Sometimes, we need to do **high-throughput** calculations(which means we need to calculate a series of temperature, pressure points for multiple phases).

It would be a great burden for users to execute these tasks manually and monitor the tasks' execution.

We provide the workflow tools based on apache-airflow workflow framework.

>we refer this docs [airflow official docs](https://airflow.apache.org/docs/apache-airflow/stable/index.html) for more instructions.


### TI_Workflow
We implement a workflow de
implemented at `workflow/DpFreeEnergy.py`

example dir and json:
```
cd examples/
cat examples/FreeEnergy.json
```

Requirement: setup apache-airflow or use the docker version dpti
```
docker run --name dpti -p 9999:8080 -it deepmodeling/dpti:latest /bin/bash
docker exec -it dpti /bin/bash
```
Then we provide a basic example for apache-airflow usage

```
# pwd at /home/airflow/
cd dpti/examples/
airflow dags trigger  TI_taskflow  --conf $(printf "%s" $(cat FreeEnergy.json))
```



**Input:** Lammps structure file.

**Output:** free energy values at given temperature and pressure.

**Parameters:** Given temperature(or the range), Given pressure(or therange), force field,Lammps simulation ensemble and etc.

we implement a workflow called TI_taskflow:
It includes these steps:
1. npt simulation to get lattice constant.
2. nvt simulation.
3. HTI: free energy at given temperature and pressure
4. TI: free energy values at the given range of temperature/pressure.

website to manage and monitor these jobs.


<a name="ac613c1818ba355261da25b7b1a1e194"></a>
# 📌Calculation Results and files

<br />For each step, the result files are located at the corresponding location.<br />For example, we start a calculation at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t`<br />
<br />For NPT MD simulation, the result file will  locate at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t/NPT_sim/result`<br />
<br />For TI simulation the result will locate at `/home/fengbo/4_Sn/1_free_energy/400K-0bar-bct-t/TI_sim/result`<br />
<br />You may want to use the result file and datas of  TI_sim/result and plot the free_energy vs T curve for different structure and find the crossing point.<br />

For HTI simulation the result will locate at `/home/fengbo/4_Sn/1_free_energy/400K-0bar-bct-t/HTI_sim/result`, a pure txt file.

The hti out file at
`new_job/02.spring_off/hti.out` which records the integration node on the path is also helpful.

<a name="ce1a1845ed13daf90d6c7ab2e135f1e0"></a>
# 💎The procedure of the free energy calculations

<br />To calculate out Gibbs (or Helmholtz) free energy of the materials, there are four steps.<br />

1. NPT MD simulation
2. NVT MD simulation
3. Hamiltonian thermodynamic integration
4. thermodynamic integration


## FreeEnergy.json

<br />FreeEnergy calculation settings for solid

| Field | Type | Example | Discription |
| --- | --- | --- | --- |
| target_temp | positive integer | 200 | the temperature of HTI |
| target_press | non-negative integer | 50000 | unit :bar.the pressure of HTI |
| work_base_dir | string | "/home/user1/metal_Sn" | see note1. work  directory.  |
| ti_path | "t" or "p" | "t" | thermodynamic integration along temperature or pressure |
| conf_lmp | string | "bct.lmp" | see note1. the materials structure to be calculated |
| ens | string | "npt-iso" | MD simulation ensemble in lammps |
| if_liquid | bool | false | if simulate liquid |

