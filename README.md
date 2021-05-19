deepmodeling, deepmd, dpti, free energy, phase diagram

---

github:  [https://github.com/deepmodeling/dpti](https://github.com/deepmodeling/dpti)<br />relating docs: [https://www.yuque.com/dpti/manual/ilzmlb](https://www.yuque.com/dpti/manual/ilzmlb)
<a name="0487c87d66ac0af8f7df818b7e010bd0"></a>
# 🌾OutPut show

We could use dpti to calculate out the Press-Volume phase diagram of metals.<br />The picture below shows the metal Sn phase diagram results calculated by one of the authors.


<br />The left subgraph shows the experiment phase diagram results.(see:[https://aip.scitation.org/doi/10.1063/1.4872458](https://aip.scitation.org/doi/10.1063/1.4872458))<br />
<br />The middle subgraph shows the DP phase diagram based on SCAN functional DFT calculation results.<br />
<br />The right subgraph shows the DP phase diagram base on PBE functional DFT calculation results.<br />
<br />![相图VASP.png](https://cdn.nlark.com/yuque/0/2021/png/3004239/1617780579997-c2f8b233-9792-4f4d-a98f-73ea4f10e178.png#height=864&id=K6sJl&margin=%5Bobject%20Object%5D&name=%E7%9B%B8%E5%9B%BEVASP.png&originHeight=864&originWidth=1728&originalType=binary&size=155958&status=done&style=none&width=1728)<br />
<br />

<a name="ad44045ba5f9b5ebf81388ff611d8d5b"></a>
# 🏞brief Introduction

**dpti** (deep potential thermodynamic integration) is a python package for calculating free energy, doing thermodynamic integration and figuring out pressure-temperature phase diagram for materials with molecular dynamics (MD) simulation methods.


<br />The user will get Gibbs (Helmholtz) free energy of a system at different temperature and pressure conditions. With these free energy results, the user could determine the phase transition points and coexistence curve on the pressure-volume phase diagram. 
<a name="xuFE2"></a>
# 🦴software introduction
At first, dpti is a collection of python scripts to generate LAMMPS input scripts and to anaylze results from LAMMPS logs.<br />
<br />In dpti, there are many MD simulations tasks and scripts need to be run sequentially or concurrently. Before and after these MD simulation tasks, we may run a lot of MD scirpts to prepare the input files or analyze the logs to extract the useful data.<br />
<br />Then the dpti developers use apache-airflow to resolve the MD tasks dependencies and managing running tasks. <br />

<a name="46bdda688b5bc33d261bccfb389fdf55"></a>
# 📃Installation

dpti use apache-airflow as workflow framework, and dpdispatcher to interact with the HPC systems (slurm or PBS).


airflow use realation database (PostgreSQL, MySQL or Sqlite) as backend to store the metadata, DAGs definetion and nodes state etc.

<a name="d3066b89f26f2ffcef7d0f8647512881"></a>
### install dpti and dpdispatcher.
git clone the following packages and install.<br />[https://github.com/deepmodeling/dpdispatcher](https://github.com/deepmodeling/dpdispatcher)<br />[https://github.com/deepmodeling/dpti](https://github.com/deepmodeling/dpti)
```bash
cd dpti/
python setup.py install

cd dpdispatcher/
python setup.py install
```


<a name="dcbb10cafb9005833579166f3acad127"></a>
###  configure apache-airflow.
airflow user manual: [https://airflow.apache.org/docs/apache-airflow/stable/index.html](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
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
    
 # you will be requested to enter the password here.
 
 
 # start airflow's webserver to manage your workflow use "-D" option to daemon it
 airflow webserver --port 8080 --hostname 127.0.0.1
 
 # start airflwo scheduler
 airflow scheduler
 
 # if ariflow web server start at the personal computer,
 # you could go to http://localhost:8080/ to view it
 # if airflow runs on remote server 
 # you could use ssh to conntect to server
 # ssh -CqTnN -L localhost:8080:localhost:8080 someusername@39.xx.84.xx
```


<a name="1f33d89b89d0c8f710b7496190e86666"></a>
# 🚀Quick Start


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
airflow's introduction on how to set up database backend: [https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html)
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

```python
# change the following item with the sql above
# sql_alchemy_conn = sqlite:////home/fengbo/airflow/airflow.db
# sql_alchemy_conn = postgres://airflow:airflow@localhost:5432/airflow
sql_alchemy_conn = postgresql+psycopg2://<user>:<password>@<host>:<port>/<db_name>
```
<a name="Yh8QG"></a>
### configure a
reset db and webserver scheduler
```python
# reset db
airflow db init
# -D flag represent daemonize
airflow webserver # -p 8080 -D
airflow scheduler # -D
```
<a name="QsNiC"></a>
### airflow webserver

TODO

<br />

<a name="253840892aedd2058f97c95ac6ef6366"></a>
# 💪extra info

The backend of this software is based on the software `airflow`. The following command can start the calculation.

<br />
<br />The first command is used for calculate the free energy of solid.<br />The second command is used for calculate the free energy of liquid.<br />

```bash
airflow trigger_dag HTI_taskflow --conf $(printf "%s" $(cat FreeEnergy.json))
airflow trigger_dag TI_taskflow --conf $(printf "%s" $(cat FreeEnergy.meam.json))
```


1. We usually want to calculate the free energy of a metal at a specific pressure or temperature.  And the crystal structure of the metal can be various. For example,  we want to calculate the free energy of metal Sn of bcc structure at 200 K and 50000 bar (5GPa). In order to caculate the per atom free energy of metal Sn. First, We must prepare a configuration file named bcc.lmp and modify the [FreeEnergy.json](#ULX0o) or [FreeEnergyLiquid.json](#WuLBQ) and modify the key-value pair  like "structure": "bcc", "target_temp": 200, "target_press" : 50000.  And decide  whether to integrate along the  t(temperature) path  or along the p(pressure) path . Modify the "path" key-value pair for this.  The key-value pair "ensemble" for lammps MD simulation. Usually the ensemble shoule be consistent with the crystal intrinsic structure. That means we should set "npt-iso" for structure "bcc" to keep the simulation box changes simultaneously in x, y, z directions.
2. modify the ti.t.json or ti.p.json, and change the key-value pair "temps" or "press" .  For ti.t.json, the tar_temp of FreeEnergy.json must be in the list  which the key-value pair "temps" of ti.t.json represents. And similarly for ti.p.json, the tar_press of FreeEnergy.json must be in the list which the key-value pair "temps" of ti.t.json represents.
3. Use the command `airflow trigger_dag`  mentioned above. This command will start a [airflow dag](https://airflow.apache.org/docs/apache-airflow/stable/concepts.html).This dag is wrote and maintained by the dpti software developer. It is used to make the calculation to be done more autocally . The user could monitor the task state and calculation procedure at [a website](#2aabcbd6). The user can also rerun, restart, delete the whole calculation or some part of the calculations.
4. Wait until the calculation finish. Usually the whole procedure continues for about 6 to 10 hours. The calculations will be done autocally.
5. Find the results in [Results Show](#2aabcbd6) part. The user could use the tables and data of it and plot the curve.



<a name="ac613c1818ba355261da25b7b1a1e194"></a>
# 📌Results Show

<br />For each step, the result files are located at the corresponding location.<br />For example, we start a calculation at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t`<br />
<br />For NPT MD simulation, the result file will  locate at `/home/user1/metal_Sn/1_free_energy/400K-0bar-bcc-t/NPT_sim/result`<br />
<br />For TI simulation the result will locate at `/home/fengbo/4_Sn/1_free_energy/400K-0bar-bct-t/TI_sim/result`<br />
<br />You may want to use the result file and datas of  TI_sim/result and plot the free_energy vs T curve for different structure and find the crossing point.<br />

The user can ssh contect to the aliyun cloud server by command like ` ssh -L localhost:8080:localhost:8080 user1@67.xxx.xxx.25`  and visit [http://localhost:8080/](http://localhost:8080/) to monitor the free energy calculation tasks procedure.


<br />
<br />

<a name="ce1a1845ed13daf90d6c7ab2e135f1e0"></a>
# 💎The procedure of the free energy calculations

<br />To calculate out Gibbs (or Helmholtz) free energy of the materials, there are four steps.<br />

1. NPT MD simulation  
1. NVT MD simulation
1. Hamiltonian thermodynamic integration
4. thermodynamic integration



<a name="63776a50b2f742b9a30748457523f601_h2_0"></a>
## NPT simulation
Run a long MD simulation and we will get the lattice constant and the best simulation box for the simulations next.
<a name="39392588f286118c98731051706b06dc_h2_1"></a>
## NVT simulation
Run a long MD simulation with the end structure of NPT simulations. We will know whether the box is reasonable enough from the results of this MD simulation.
<a name="14932184266288e8651c16aebdbfbb8e_h2_2"></a>
## Hamiltonian thermodynamic integration (HTI)
We will know the Gibbs (or Helmholtz) free energy at the specific temperature or pressure condition. 
<a name="56937698a8d18e2013bea45f6dfe5890_h2_3"></a>
## thermodynamic integration (TI)
Integrating along the isothermal or isobaric path, We will know the free energy at different pressure and temperature.<br />

<a name="ebf132332b4d6dc4e1ac472cd3ca6183_h1_5"></a>
# **🌾**JSON file settings

<br />There are diffefrent json files desinged for different usage.<br />

1. FreeEnergy.json   control the whole workflow
1. npt.json
3. nvt.json
3. hti.json or hti.liquid.json
3. ti.t.json or ti.p.json
<a name="bec45b57cca51fd87d4476957f294010"></a>
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




note:

1. the conf_lmp file must be in the work_base_dir.


<br />

<a name="4f482566dea51cf942c738f10e1019e6"></a>
## npt.json

<br />the settings used in MD NPT simulations.

| Field | Type | Example | Description |
| --- | --- | --- | --- |
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



<a name="b25f2fcec09423c879e1d1f7c34f497d"></a>
## nvt.json
the settings used in MD NPT simulations

| Field | Type | Example | Description |
| --- | --- | --- | --- |
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



<a name="245f4b95f29b388af105193562a72472"></a>
## hti.json
For solid, the settings used in Hamiltonian thermodyniamics integration (HTI)

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| ncopies | list of integer | [1,1,1] | do not change this pair |
| lambda_lj_on | list of arange | ["0.000:0.100:0.0125",<br />"0.100:0.200:0.025",<br />"0.200:1.000:0.2",<br />"1"] | the lambda value used in 00.lj_on numerial integration |
| lambda_deep_on | list of arange  | ["0.00:0.05:0.010",<br />"0.05:0.15:0.02",<br />"0.15:0.35:0.040",<br />"0.35:1.00:0.065",<br />"1"] | the lambda value used in 01.deep_on numerial integration |
| lambda_spring_off | list of arange  | ["0.000:0.750:0.125",<br />"0.750:0.900:0.050",<br />"0.900:0.960:0.020",<br />"0.960:1.00:0.010",<br />"1"] | the lambda value used in 02.spring_off numerial integration |
| protect_eps | float (usuall small positive number) | 1e-06 | the minimum lambda number used in numerial integration |
| model | string  | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atomic mass |
| spring_k | float | 0.02 | spring constant used in Einstein solid. |
| soft_param | dictionary | {"sigma_0_0":2,7,<br />"epsilon":0.030,<br />"activation":0.5,<br />"n":1.0,<br />"alpha_lj":0.5,<br />"rcut":6.0} | see: note1 below |
| crystal | "frenkel" or "vega | "frenkel" | different  Einstein solid approximation method |
| langevin |  bool |  true | whether use langevin  thermostat  |
| nsteps | integer | 200000 | MD steps in each simulation |
| timestep | float | 0.002 | time_step in lammps MD simulation (unit: picosecond) |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 10000 | skip the first n steps in statistic  |
| stat_bsize | integer | 200 | batch size in statistic |
| temp | integer | 400 | the target temperature in HTI calculation  |


note:

1. the parameter defined by lammps pair_style  lj/cut/soft and pair_coeff command. see [lammps lj/cut/soft](https://lammps.sandia.gov/doc/pair_fep_soft.html#)
2. sigma_0_0 means the sigma value for the lammps atom type 0 and atom type 0.

<a name="379b68fde55209594cf49f462acad39b"></a>
## hti.liquid.json
For solid, the settings used in Hamiltonian thermodyniamics integration (HTI).

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| ncopies | list of integer | [1,1,1] | do not change this pair |
| lambda_soft_on | list of arange | ["0.000:0.030:0.003",<br />"0.030:0.060:0.005",<br />"0.060:0.160:0.010",<br />"0.160:0.300:0.020",<br />"0.300:1.000:0.050",<br />"1"] | the lambda value used in 00.soft_on numerial integration |
| lambda_deep_on | list of arange  | ["0.000:0.006:0.002",<br />"0.006:0.030:0.004",<br />"0.030:0.100:0.010",<br />"0.100:0.400:0.030",<br />"0.400:1.000:0.060",<br />"1"] | the lambda value used in 01.deep_on numerial integration |
| lambda_soft_off | list of arange  | ["0.000:0.750:0.125",<br />"0.750:0.900:0.050",<br />"0.900:1.000:0.020",<br />"1"] | the lambda value used in 02.soft_off numerial integration |
| protect_eps | float (usually small positive number) | 1e-06 | the minimum lambda number used in numerial integration |
| model | string  | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atomic mass |
| spring_k | float | 0.02 | spring constant used in Einstein solid. |
| soft_param | dictionary | {"sigma_0_0":2,7,<br />"epsilon":0.030,<br />"activation":0.5,<br />"n":1.0,<br />"alpha_lj":0.5,<br />"rcut":6.0} | see: note1 below |
| crystal | "frenkel" or "vega | "frenkel" | different  Einstein solid approximation method |
| langevin |  bool |  true | whether use langevin  thermostat  |
| nsteps | integer | 200000 | MD steps in each simulation |
| timestep | float | 0.002 | time_step in lammps MD simulation (unit: picosecond) |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 10000 | skip the first n steps in statistic  |
| stat_bsize | integer | 200 | batch size in statistic |
| temp | integer | 400 | the target temperature in HTI calculation  |


note:

1. the parameter defined by lammps pair_style  lj/cut/soft and pair_coeff command. see [lammps lj/cut/soft](https://lammps.sandia.gov/doc/pair_fep_soft.html#)
2. sigma_0_0 means the sigma value for the lammps atom type 0 and atom type 0.

<a name="39d1080db8e8cc133d75af64f089abd5"></a>
## ti.t.json
the settings used in thermodynamic integration (TI) for constant pressure and changeable temperature

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| copies | list of integer | [1,1,1] | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 200000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | npt-aniso | lammps MD simulation ensemble setting |
| path | "t" or "p" | "t" | do not change this pair for ti.t.json |
| temp_seq | list of arange | ["200:1400:20",<br />1400] | temperature list to be calculated. The HTI tar_temp must be in it. |
| pres |  integer | 50000 | the target pressure of HTI calculation |
| tau_t | float | 0.2 | lammps Tdamp |
| tau_p | float  | 2.0 | lammps Pdamp |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 5000 | skip the first n steps in statistic |
| stat_bsize | integer | 200 | statistic batch size |

<a name="f3b978108de979f9529b07f01b19dc5d"></a>
# 
<a name="cf0461961f6d77fdee29d68f2bc9982a"></a>
## ti.p.json
the settings used in thermodynamic integration (TI) for constant temperature and changeable  pressure

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| equi_conf | string | "conf.lmp" | do not change this pair |
| copies | list of integer | [1,1,1] | do not change this pair |
| model | string | "graph.pb" | do not change this pair |
| mass_map | list of float | [118.71] | relative atom mass |
| nstep | integer | 200000 | MD simulation steps in the lammps NPT simulation |
| timestep | float | 0.002 | lammps script time_step. unit: picosecond |
| ens | string | npt-aniso | lammps MD simulation ensemble setting |
| path | "t" or "p" | "t" | do not change this pair for ti.t.json |
| temp | integer | 800 | temperature to be calculated |
| pres_seq | list of arange | [0:100000:2000,<br />100000] | the pressure list to be calculated. The HTI tar_pres must be in it. |
| tau_t | float | 0.2 | lammps Tdamp |
| tau_p | float  | 2.0 | lammps Pdamp |
| thermo_freq | integer | 10 | statistic frequency |
| stat_skip | integer | 5000 | skip the first n steps in statistic |
| stat_bsize | integer | 200 | statistic batch size |

<a name="bKzHP"></a>
## gdi.json
The gdi.json is used for gibbs-duham integration.  When you know the one point at the two phase coexisting-line. You can do gibbs-duham integration to get the whole phase boundry

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| phase_i | dict | {"name": "PHASE_0",<br />"equi_conf":"bct.lmp",<br />"ens":"npt-xy"} | phase 1 information<br /> |
| phase_ii | dict | {"name": "PHASE_1",<br />"equi_conf":"liquid.lmp",<br />"ens":"npt-iso"} | phase 2 information |
| model | str | "graph.pb" |  |
| mass_map | list of float | [118.71] | relative atomic mass |
| nsteps | integer | 100000 | MD steps in simulation |
| timestep | float | 0.002 | MD timestep (in ps) |
| tau_t | float | 0.1 | MD NPT tau_t |
| tau_p | float  | 1.0 | MD NPT tau_p |
| thermo_freq | integer | 10 | MD thermo frequency |
| stat_skip | integer | 5000 | skip the first 5000 steps in lammps log |
| stat_bsize | integer | 100 | statistic batch size |



<a name="a06a67f0f7f8127369b77b7736067707"></a>
# **👀**Troubleshooting
TODO<br />
<br />

