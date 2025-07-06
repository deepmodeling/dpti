# docker pull yfb222333/dpti-lammps-fep:latest 
#  docker run -it --gpus all dpti-lammps-fep:latest
# all in /opt/
# all files in /opt/dpti/examples/water_SCAN/ice04/

#NPT
dpti equi gen npt.json -o npt/

# submit jobs `lmp -i in.lammps` for npt/

dpti equi extract ./npt/ -o npt_avg.lmp # extract average

# NVT
dpti equi gen nvt.json -o nvt/

# submit jobs `lmp -i in.lammps` for nvt/

dpti equi extract ./nvt/ -o nvt_last_dump.lmp # extract average

## HTI (hti,hti_liq, hti_water, hti_ice) module
dpti hti_ice gen hti_ice.json -s three-step -o hti/

# submit jobs `lmp -i in.lammps` for subdirs like  `hti/0*/task*/`

dpti hti_ice compute  ./hti/  -t gibbs --npt ./npt/  # note use NPT simulation usually longer steps, and the P*V value is more accurate  

# result txt in ./hti/result

## TI (ti, ti_water) module

dpti ti_water gen path-t.json -o ti_path_t/

# submit jobs `lmp -i in.lammps` for subdirs like  `ti_path_t/task*/`

 dpti ti_water compute ./ti_path_t/  --hti ./hti/  # HTI simulation (starting point, gibbs free energy value)

#  final result for a ti-line txt in ./ti_path_t/result




