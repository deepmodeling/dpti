# introduction:
This file indicate the calculation example for calculating Sn Beta phase High Pressure Phase Diagram.
See: https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.7.053603


The files in this dir describe the relating json files for HTI calculating Tin's Free Energy at 200K,2GPa.

And then perform TI calculations with starting point 200K,2GPa and temperature range from 200K to 1800K with interval 100K (at 2GPa).


The example LAMMPS simulation MD steps:

NPT: 1m steps. NVT:200k steps. TI: 300k steps  HTI 500k for each lambda value.

This example can be used as a template. The parameters in each json file could be changed.
