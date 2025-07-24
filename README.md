deepmodeling, deepmd, dpti, free energy, phase diagram

---

# 🏞brief Introduction

**dpti** (deep potential thermodynamic integration) is a python package for calculating free energy, doing thermodynamic integration and figuring out pressure-temperature phase diagram for materials with molecular dynamics (MD) simulation methods.


<br />The user will get Gibbs (Helmholtz) free energy of a system at different temperature and pressure conditions. With these free energy results, the user could determine the phase transition points and coexistence curve on the pressure-volume phase diagram.
<a name="xuFE2"></a>

useful links:

github README.md:  [https://github.com/deepmodeling/dpti/README.md](https://github.com/deepmodeling/dpti/README.md)

On Bohrium Platform:
Bohrium notebook:
The basic usage of dpti, commands and examples:
https://www.bohrium.com/notebooks/82544159178

Demo site:
https://www.bohrium.com/apps/dpti


intruction to free energy calculation via thermodynamic integration method:
[https://nb.bohrium.dp.tech/detail/18465833825](https://nb.bohrium.dp.tech/detail/18465833825)



<a name="0487c87d66ac0af8f7df818b7e010bd0"></a>
# 🌾OutPut show

## for water and ice

see: [PRL:Phase Diagram of a Deep Potential Water Model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.236001)

Phase diagram of water. DP model (red solid lines) and experiment (gray solid lines) for $T<420K$. Black letters indicate phases that are stable in the experiment and model. The original figure is in article。

![water_phase_diagram.png](https://journals.aps.org/prl/article/10.1103/PhysRevLett.126.236001/figures/1/medium)


## for metal Tin(Sn)
We could use dpti to calculate out the Press-Volume phase diagram of metals.<br />The picture below shows the metal Sn phase diagram results calculated by one of the authors.


<br />The left subgraph shows the experiment phase diagram results.(see:[https://aip.scitation.org/doi/10.1063/1.4872458](https://aip.scitation.org/doi/10.1063/1.4872458))<br />
<br />The middle subgraph shows the DP phase diagram based on SCAN functional DFT calculation results.<br />
<br />The right subgraph shows the DP phase diagram base on PBE functional DFT calculation results.<br />
<br />![相图VASP.png](https://cdn.nlark.com/yuque/0/2021/png/3004239/1617780579997-c2f8b233-9792-4f4d-a98f-73ea4f10e178.png#height=864&id=K6sJl&margin=%5Bobject%20Object%5D&name=%E7%9B%B8%E5%9B%BEVASP.png&originHeight=864&originWidth=1728&originalType=binary&size=155958&status=done&style=none&width=1728)<br />
<br />

<a name="ad44045ba5f9b5ebf81388ff611d8d5b"></a>

# 🦴software introduction
At first, dpti is a collection of python scripts to generate LAMMPS input scripts and to anaylze results from LAMMPS logs.<br />
<br />In dpti, there are many MD simulations tasks and scripts need to be run sequentially or concurrently. Before and after these MD simulation tasks, we may run a lot of MD scirpts to prepare the input files or analyze the logs to extract the useful data.<br />
<br />Then the dpti developers use apache-airflow to resolve the MD tasks dependencies and managing running tasks. <br />

# directory structure

useful directories:


# 
```
dpti/ # main directory for dpti python modules
dpti/lib/ usefule libs for numberical integration and analysis, free energy calculation, etc.

# example dir structure

examples/ # examples dir
examples/Sn_High_Pressure/ # example for Sn high pressure phase diagram
examples/water_SCAN/ # example for water phase diagram

# developer tools

tests/ # unittests
docker/ # dockerfile
conda/ conda release
```


# Software Usage:

>the examples dir `examples/` in source code contains the essential files and jsons.

## for CLI tools:
The following scripts can be used by Python CLI to generate essential scripts for LAMMPS simulation.

the CLI entry:

```
# after installation: pip install .
dpti --help
```

```
# example dir:
cd examples/Sn_High_Pressure/

#download the models:
wget https://huggingface.co/Felix5572/Sn-SCAN-Compressed/resolve/main/graph.pb -O Sn_SCAN_compressed.pb
```

general useful commands:

```
# NPT
dpti equi gen npt.json -e npt-xy -t 200 -p 20000  -o NPT_sim/
# cd NPT_sim/new_job/
dpti equi extract ./  -o npt_avg.lmp

# NVT
dpti equi gen equi_settings.json --ensemble nvt -t 200 -p 20000 --conf-npt ./NPT_sim/new_job/ -o NVT_sim/
# cd NVT_sim/new_job/
dpti equi extract ./  -o nvt_last_dump.lmp

# HTI
dpti hti gen hti.json -s three-step -o HTI_sim/
# dpti hti_water gen hti_water.json -o HTI_water/
# dpti hti_ice gen hti_ice.json  -s three-step -o HTI_ice/
dpti hti compute ./new_job/  -t gibbs --npt ../NPT_sim/new_job/

# TI
dpti ti gen ti_settings.json -o TI_sim/
dpti ti compute ./TI_sim/new_job/  --hti ../HTI_sim/new_job/
```



### Equi(npt and nvt simulation)
The following scripts are used to generate essential tools.

```
cd exampls/equi/
dpti equi --help`
dpti equi gen npt.json
dpti equi gen nvt.json
```
The dir `new_job/` contains the simulation files


### HTI

This is an example for HTI three-step simulations.

```
cd examples/hti/
dpti hti gen hti.json -s three-step -o HTI_sim/
# dpti hti_water gen hti_water.json -o HTI_water/
# dpti hti_ice gen hti_ice.json  -s three-step -o HTI_ice/

# After the lammps simulations.
dpti hti compute ./new_job/  -t gibbs --npt ../NPT_sim/new_job/
```

### TI

For temperature 200K, in order to generate integration path pressure from 0 to 10000 of interval 500.

in `ti.p.json`, we writes
```json
"temp":200,
"pres_seq":[
    "0:10000:500", 
    "10000"
]
```

```
cd examples/ti/
dpti ti gen ti.t.json
```

In order to generate TI path changing temperature, we use
```
dpti ti gen ti.p.json
```

After the TI simulations. Calculate the free energy values lines with the corresponding HTI results.
```
dpti ti compute ./TI_sim/new_job/  --hti ../HTI_sim/new_job/
```


### GDI
An example for finding coexisting line between Sn `beta` and `alpha` phase.

by `gdidata.json`:
starting point is 1GPa,270K. (calculated by HTI method)
We want to extend to 1.35 GPa.

```
cd examples/gdi/
dpti gdi pb.json machine.json -g gdidata.json
```

``


<a name="46bdda688b5bc33d261bccfb389fdf55"></a>
# 📃Installation


## local CLI tools installtion


```
# usually create a new python environment
# conda create --name dpti
# conda activate dpti
cd dpti/
pip install .
# use this command to check installation
dpti --help
```

## docker image:
```
docker pull yfb222333/dpti-lammps-fep:latest
```

On Bohrium Platform, Image address:
https://www.bohrium.com/image/detail/dpti/dpti


## Manually installation

>the [Dockerfile](docker/Dockerfile) at `docker/` dir may be helpful for mannually installation.


<a name="d3066b89f26f2ffcef7d0f8647512881"></a>
### install dpti and dpdispatcher.
git clone the following packages and install.<br />
[https://github.com/deepmodeling/dpti](https://github.com/deepmodeling/dpti)
```bash
cd dpti/
pip install .
```

Note: dpti's lammps command requires Lammps package to be installed. (with USER-FEP and USER-DEEPMD packages enabled)





<!-- ### install postgresql backend

apahche-airflow require a database backend.Here we refer to this doc [postgresql offical docs for download](https://www.postgresql.org/download/)

and use this command
```
psql -h
```


<a name="dcbb10cafb9005833579166f3acad127"></a>
###  configure apache-airflow.
airflow user manual: [https://airflow.apache.org/docs/apache-airflow/stable/index.html](https://airflow.apache.org/docs/apache-airflow/stable/index.html)
```bash
# airflow will create at ~/airflow
airflow -h
cd ~/airflow

# usually the configuration file location
# we refer this doc for further information
# https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html
vi ~/airflow/airflow.cfg

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
``` -->


<a name="1f33d89b89d0c8f710b7496190e86666"></a>
# 🚀Quick Start

## with docker
```docker pull yfb222333/dpti-lammps-fep:latest```

Image usage:

```
# Usage Instructions:
# 1. Run container:
#    docker run -it --gpus all dpti-lammps-fep

# 2. (in container)Test commands:
#    dp -h
#    lmp -h
#    cd deepmd-fep-testcase/ && lmp -i in.lammps

# 3. Build commands:
#    # Build base image
#    docker build --target dpti-devel-base --tag dpti-devel-base:latest -f dp-lammps-fep.Dockerfile ./
   
#    # Build final image (two methods):
#    # Method 1: Direct build
#    docker build --tag dpti-lammps-fep:latest -f dp-lammps-fep.Dockerfile ./
#    # Method 2: Build with cache from base
#    docker build --cache-from dpti-devel-base:latest --tag dpti-lammps-fep:latest -f dp-lammps-fep.Dockerfile ./
```

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

1. npt.json
2. nvt.json
3. hti.json or hti.liquid.json
4. ti.t.json or ti.p.json
<a name="bec45b57cca51fd87d4476957f294010"></a>



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
| thermo_freq | integer | 10 | statistic frequency(lammps keywork `thermo`) |
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
| stat_skip | integer | 5000 | skip the first 5000 value in lammps log |
| stat_bsize | integer | 100 | statistic batch size |

##  thermo_freq, stat_bsize, stat_skip example:
if we set nsteps==500000, thermo_freq==10, stat_bsize==100, stat_skip==1000,
then first we will run a 500000 steps MD, and generate 500000/10==50000 data points.
the first stat_skip==1000 points will be ignored,
then the last 49000 points will be grouped into 49000/100==490 chunks.
The average value the of the 100 point value in each chunk will treated as chunk value.
And the 490 chunk values will also be averaged as the final result value for this MD.

### pb.json

example command:

```python gdi.py pb.josn machine.json -g gdidata.json```

This file is used by `dpti gdi` module

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| equi_conf | string | "beta.lmp" | target simulation temperature unit:K |
| ens | choice: "npt-aniso" "npt-iso" "npt-xy" | "npt-xy" | simulation ensemble |
| model | string | "../graph.pb" | model path |

### gdidata.json

This file is used to specify the values GDI (phase coexisting line simulation)

This example shows begining phase transition point at 1GPa,270K. And we want to extend to 1.35GPa .


>Note: when calculating, the solver keeps the local error estimates less than `atol + rtol * abs(y)`. See:[scipy.integrate.solve_ivp doc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

Note: if `direction` defines GDI:calculation behaviour.
"p": changing pressure from `begin` to `end` with initial temperature `initial_value`.

"t": changing temperature from `begin` to `end` with initial pressure `initial_value`.

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| begin | integer | 10000 | begin simulation temperature or pressure. unit:K/Bar |
| end | integer | 13500 | end simulation temperature or pressure. unit:K/Bar |
| initial_value | integer | 270 | given initial value |
| abs_tol | float | 2 | used by scipy solve_ivp function |
| rel_tol | float | 0.001 | used by scipy solve_ivp function. |
| direction | choice:"t" "p"| "p" | see above |

# FAQ

## simulation related:

### phase transition:
For thermodynamics integration: there is a restrict that no phase transition should happen during the integration path.
This means that for the simulation process lambda changing from 0 to 1, the structure of system should not change dramatically. (like metal to liquid, lattice constants largely change).

In practice, we could monitor the RDF(compute rdf) of the structure during simulation. The rdf could be used as an indicator for the structure changes. The RMSD value of this system could also be used as indicator.

### integration path:

For thermodynamics integration, sometimes it may not be a choice to directly change from the initial state(Einstein solid, Ideal Gas) to the final target state.

To extend the integration and avoid phase transition during MD simulation, it may be better to introduce some intermediate during simulation, this is implemented in dpti software called two-steps and three-steps.(At least, this strategy could take effects for water(ice) and metal Tin(Sn))

For researcher, it is recommended to try both the direct path protocol and the intermediate state protocol. And compare the results.
And repeat the calculation at least for one more time at a specific temperature and pressure and check the result consistence.

If the result errors lie in about 1meV/per atom. (maybe about 10K-20K in phase diagram) We could treat it as a reliable result.

### atom number in simulation:
Usually the atom number should be about 100-200. Larger system is OK.
Size effect is not obvious.(increasing the simulation size will usually get similar free energy values).
