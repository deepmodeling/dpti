deepti is used to calcute free energy (and melting point or related property) of metals and water with molecular dynamics simulation methods

lateset user manual and installation guide
see: https://www.yuque.com/deepti/manual/ilzmlb

deepmodeling, deepmd, deepti, free energy, phase diagram

---

# 🌾OutPut show
deep potential metal Sn. experiment phase diagram. SCAN functional model.  PBE functional model.  Press-Volume phase diagram.
![相图VASP.png](https://cdn.nlark.com/yuque/0/2021/png/3004239/1617780579997-c2f8b233-9792-4f4d-a98f-73ea4f10e178.png#align=left&display=inline&height=864&margin=%5Bobject%20Object%5D&name=%E7%9B%B8%E5%9B%BEVASP.png&originHeight=864&originWidth=1728&size=155958&status=done&style=none&width=1728)
**
**
# 🏞Introduction
:::info
**dpti** (deep potential thermodynamic integration) is a python package for calculating free energy, doing thermodynamic integration and figure out pressure-temperature phase diagram for materials with molecular dynamics (MD) simulation method.
:::


The user will get Gibbs (Helmholtz) free energy at different temperature and pressure conditions. And the user can use the results to figure out the 


# 📃Installation
:::info
deepti use apache-airflow as job scheduler , and dpdispatcher as execute MD tasks, and airflow may need postgresql as database.
:::
### install deepti and dpdispatcher.
git clone the following packages and install.
[https://github.com/deepmodeling/dpdispatcher](https://github.com/deepmodeling/dpdispatcher)
[https://github.com/deepmodeling/dpti](https://github.com/deepmodeling/dpti)
```bash
cd dpti
python setup.py install

cd dpdispatcher
python setup.py install
```


###  configure apache-airflow.
airflow user manual [https://airflow.apache.org/docs/apache-airflow/stable/index.html](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
```bash
# airflow will create at ~/airflow
airflow -h
cd ~/airflow

# airflow will initialize datebase with sqlite
airflow db init

# create a user
airflow users create \
    --username airflow \
    --firstname Peter \
    --lastname Parker \
    --role Admin \
    --email spiderman@superhero.org
    
 # you will be requested to enter the password.
 
 
 # start airflow's webserver to manage your workflow use "-D" option to daemon it
 airflow webserver --port 8080
 
 # start airflwo scheduler
 airflow scheduler
 
 # if ariflow web server start at the personal computer,
 # you could go to http://localhost:8080/ to view it
 # if airflow runs on remote server 
 # you could use ssh to conntect to server
 # ssh -CqTnN -L localhost:8080:localhost:8080 someusername@39.xx.84.xx
```
# 


# 🚀Quick Start
Once we start airflow scheduler . The ariflow can receive job.
```bash
 # copy deepti'workflow file  
 cp /path-to-deepti/workflow/DpFreeEnergy.json ~/airflow/dags/
 
 # create a workdir and copy example files
 cp /path-to-deepti/examples/* /path-to-a-work-dir/
 
 # start our airflow job
 cd /path-to-a-work-dir/
 cat ./airflow.sh
 
 airflow dags trigger  TI_taskflow  --conf $(printf "%s" $(cat FreeEnergy.json))
 
```


##  
## 🕹install postgresql database
## 

The default sqlite database of airflow can only run one task at the same time.
So we should configure postgresql.
see [https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html)
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
 postrgresql create database user 
```sql
CREATE DATABASE airflow_db;
CREATE USER airflow_user WITH PASSWORD 'airflow_user';
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;
```
configure  ~/airflow/airflow.cfg


```python
# change the following item with the sql above
# sql_alchemy_conn = sqlite:////home/fengbo/airflow/airflow.db
sql_alchemy_conn = postgresql+psycopg2://<user>:<password>@<host>/<db>
```
reset db and webserver scheduler
```python
# reset db
airflow db init
airflow webserver
airflow scheduler
```
# 💪extra info
:::warning
This project is under development now. This software is deployed on Alibaba Cloud Service of DeePMD group.
:::


:::success
The backend of this software is based on the software `airflow`. The following command can start the calculation.
:::




The first command is used for calculate the free energy of solid.
The second command is used for calculate the free energy of liquid.


```bash
airflow trigger_dag HTI_taskflow --conf $(printf "%s" $(cat FreeEnergy.json))
airflow trigger_dag TI_taskflow --conf $(printf "%s" $(cat FreeEnergy.meam.json))
```


1. To calculate the free energy of some structure at specific pressure or temperature. For example, bcc structure (bcc.lmp) at 200 K and 50000 bar (5GPa). You should modify the [FreeEnergy.json](#ULX0o) or [FreeEnergyLiquid.json](#WuLBQ) and modify the key-value pair  like "structure": "bcc", "tar_temp": 200, "tar_press" : 50000.  And decide the whether to integrate along the  t(temperature) path  or along the p(pressure) path . Modify the "path" key-value pair for this.  The key-value pair "ensemble" for lammps MD simulation. Usually the ensemble shoule be consistent with the crystal intrinsic structure. That means we should set "npt-iso" for structure "bcc" to keep the simulation box changes simultaneously in x y z directions.
1. modify the ti.t.json or ti.p.json, and change the key-value pair "temps" or "press" .  For ti.t.json, the tar_temp of FreeEnergy.json must be in the list  which the key-value pair "temps" of ti.t.json represents. And similarly for ti.p.json, the tar_press of FreeEnergy.json must be in the list which the key-value pair "temps" of ti.t.json represents.
1. Use the command `airflow trigger_dag`  mentioned above. This command will start a [airflow dag](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html).This dag is wrote and maintained by the deepti software developer. It is used to make the calculation to be done more autocally . The user could monitor the task state and calculation procedure at [a website](#2aabcbd6). The user can also rerun, restart, delete the whole calculation or some part of the calculations.
1. Wait until the calculation finish. Usually the whole procedure continues for about 6 to 10 hours. The calculations will be done autocally.
1. Find the results in [Results Show](#2aabcbd6) part. The user could use the tables and data of it and plot the curve.



# 📌Results Show


For each step, the result files are located at the corresponding location.
For example, we start a calculation at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t`


For NPT MD simulation, the result file will  locate at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t/NPT_sim/result`


For TI simulation the result will locate at `/home/fengbo/4_Sn/1_free_energy/400K-0bar-bct-t/TI_sim/result`


You may want to use the result file and datas of  TI_sim/result and plot the free_energy vs T curve for different structure and find the crossing point.


:::info
The user can ssh contect to the aliyun cloud server by command like ` ssh -L localhost:8080:localhost:8080 user1@67.xxx.xxx.25`  and visit [http://localhost:8080/](http://localhost:8080/) to monitor the free energy calculation tasks procedure.
:::






# 💎The procedure of the free energy calculations


To calculate out Gibbs (or Helmholtz) free energy of the materials, there are four steps.


1. NPT MD simulation  
1. NVT MD simulation
1. Hamiltonian thermodynamic integration
4. thermodynamic integration



## NPT simulation
Run a long MD simulation and we will get the lattice constant and the best simulation box for the simulations next.
## NVT simulation
Run a long MD simulation with the end structure of NPT simulations. We will know whether the box is reasonable enough from the results of this MD simulation.
## Hamiltonian thermodynamic integration (HTI)
We will know the Gibbs (or Helmholtz) free energy at the specific temperature or pressure condition. 
## thermodynamic integration (TI)
Integrating along the isothermal or isobaric path, We will know the free energy at different pressure and temperature.


# **🌾**JSON file settings


There are diffefrent json files desinged for different usage.


1. FreeEnergy.json   control the whole workflow
1. npt.json
3. nvt.json
3. hti.json or hti.liquid.json
3. ti.t.json or ti.p.json
## FreeEnergy.json


FreeEnergy calculation settings for solid

| Field | Type | Example | Discription |
| :--- | :--- | :--- | :--- |
| target_temp | positive integer | 200 | the temperature of HTI |
| target_press | non-negative integer | 50000 | unit :bar.the pressure of HTI |
| work_base_dir | string | "/home/user1/metal_Sn" | see note1. work  directory.  |
| ti_path | "t" or "p" | "t" | thermodynamic integration along temperature or pressure |
| conf_lmp | string | "bct.lmp" | see note1. the materials structure to be calculated |
| ens | string | "npt-iso" | MD simulation ensemble in lammps |
| if_liquid | bool | false | if simulate liquid |



:::tips
note:

1. the conf_lmp file must be in the work_base_dir.
:::




## npt.json


the settings used in MD NPT simulations.

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 1000000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | "npt-iso" | lammps MD ensemble setting |
| pres | positive integer | 50000 | pressure in MD simulation (unit:bar; 1GPa==10000bar) |
| tau_t | float | 0.2 | Tdamp in lammps fix npt command |
| tau_p | float | 2.0 | Pdamp in lammps fix npt command |
| thermo_freq | positive integer | 10 | statistic frequency |



## nvt.json
the settings used in MD NPT simulations

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 1000000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | "npt-iso" | lammps MD ensemble setting |
| pres | positive integer | 50000 | pressure in MD simulation (unit:bar; 1GPa==10000bar) |
| tau_t | float | 0.2 | Tdamp in lammps fix npt command |
| tau_p | float | 2.0 | Pdamp in lammps fix npt command |
| thermo_freq | positive integer | 10 | statistic frequency |



## hti.json
For solid, the settings used in Hamiltonian thermodyniamics integration (HTI)

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| ncopies | list of integer | [1,1,1] | do not change this pair |
| lambda_lj_on | list of arange | ["0.000:0.100:0.0125",
"0.100:0.200:0.025",
"0.200:1.000:0.2",
"1"] | the lambda value used in 00.lj_on numerial integration |
| lambda_deep_on | list of arange  | ["0.00:0.05:0.010",
"0.05:0.15:0.02",
"0.15:0.35:0.040",
"0.35:1.00:0.065",
"1"] | the lambda value used in 01.deep_on numerial integration |
| lambda_spring_off | list of arange  | ["0.000:0.750:0.125",
"0.750:0.900:0.050",
"0.900:0.960:0.020",
"0.960:1.00:0.010",
"1"] | the lambda value used in 02.spring_off numerial integration |
| protect_eps | float (usuall small positive number) | 1e-06 | the minimum lambda number used in numerial integration |
| model | string  | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atomic mass |
| spring_k | float | 0.02 | spring constant used in Einstein solid. |
| soft_param | dictionary | {"sigma_0_0":2,7,
"epsilon":0.030,
"activation":0.5,
"n":1.0,
"alpha_lj":0.5,
"rcut":6.0} | see: note1 below |
| crystal | "frenkel" or "vega | "frenkel" | different  Einstein solid approximation method |
| langevin |  bool |  true | whether use langevin  thermostat  |
| nsteps | integer | 200000 | MD steps in each simulation |
| timestep | float | 0.002 | time_step in lammps MD simulation (unit: picosecond) |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 10000 | skip the first n steps in statistic  |
| stat_bsize | integer | 200 | batch size in statistic |
| temp | integer | 400 | the target temperature in HTI calculation  |



:::tips
note:

1. the parameter defined by lammps pair_style  lj/cut/soft and pair_coeff command. see [lammps lj/cut/soft](https://lammps.sandia.gov/doc/pair_fep_soft.html#)
1. sigma_0_0 means the sigma value for the lammps atom type 0 and atom type 0.
:::
## hti.liquid.json
For solid, the settings used in Hamiltonian thermodyniamics integration (HTI).

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| ncopies | list of integer | [1,1,1] | do not change this pair |
| lambda_soft_on | list of arange | ["0.000:0.030:0.003",
"0.030:0.060:0.005",
"0.060:0.160:0.010",
"0.160:0.300:0.020",
"0.300:1.000:0.050",
"1"] | the lambda value used in 00.soft_on numerial integration |
| lambda_deep_on | list of arange  | ["0.000:0.006:0.002",
"0.006:0.030:0.004",
"0.030:0.100:0.010",
"0.100:0.400:0.030",
"0.400:1.000:0.060",
"1"] | the lambda value used in 01.deep_on numerial integration |
| lambda_soft_off | list of arange  | ["0.000:0.750:0.125",
"0.750:0.900:0.050",
"0.900:1.000:0.020",
"1"] | the lambda value used in 02.soft_off numerial integration |
| protect_eps | float (usually small positive number) | 1e-06 | the minimum lambda number used in numerial integration |
| model | string  | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atomic mass |
| spring_k | float | 0.02 | spring constant used in Einstein solid. |
| soft_param | dictionary | {"sigma_0_0":2,7,
"epsilon":0.030,
"activation":0.5,
"n":1.0,
"alpha_lj":0.5,
"rcut":6.0} | see: note1 below |
| crystal | "frenkel" or "vega | "frenkel" | different  Einstein solid approximation method |
| langevin |  bool |  true | whether use langevin  thermostat  |
| nsteps | integer | 200000 | MD steps in each simulation |
| timestep | float | 0.002 | time_step in lammps MD simulation (unit: picosecond) |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 10000 | skip the first n steps in statistic  |
| stat_bsize | integer | 200 | batch size in statistic |
| temp | integer | 400 | the target temperature in HTI calculation  |



:::tips
note:

1. the parameter defined by lammps pair_style  lj/cut/soft and pair_coeff command. see [lammps lj/cut/soft](https://lammps.sandia.gov/doc/pair_fep_soft.html#)
1. sigma_0_0 means the sigma value for the lammps atom type 0 and atom type 0.
:::
## ti.t.json
the settings used in thermodynamic integration (TI) for constant pressure and changeable temperature

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| copies | list of integer | [1,1,1] | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 200000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | npt-aniso | lammps MD simulation ensemble setting |
| path | "t" or "p" | "t" | do not change this pair for ti.t.json |
| temp_seq | list of arange | ["200:1400:20",
1400] | temperature list to be calculated. The HTI tar_temp must be in it. |
| pres |  integer | 50000 | the target pressure of HTI calculation |
| tau_t | float | 0.2 | lammps Tdamp |
| tau_p | float  | 2.0 | lammps Pdamp |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 5000 | skip the first n steps in statistic |
| stat_bsize | integer | 200 | statistic batch size |

# **
## ti.p.json
the settings used in thermodynamic integration (TI) for constant temperature and changeable  pressure

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| copies | list of integer | [1,1,1] | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 200000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | npt-aniso | lammps MD simulation ensemble setting |
| path | "t" or "p" | "t" | do not change this pair for ti.t.json |
| temp | integer | 800 | temperature to be calculated |
| pres_seq | list of arange | [0:100000:2000,
100000] | the pressure list to be calculated. The HTI tar_pres must be in it. |
| tau_t | float | 0.2 | lammps Tdamp |
| tau_p | float  | 2.0 | lammps Pdamp |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 5000 | skip the first n steps in statistic |
| stat_bsize | integer | 200 | statistic batch size |

**
# **👀**Troubleshooting
TODO

