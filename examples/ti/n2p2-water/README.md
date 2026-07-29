# n2p2 water HTI and TI example

This example demonstrates a complete free-energy path with a
Behler-Parrinello neural network potential through the LAMMPS `hdnnp` pair
style. Both Hamiltonian thermodynamic integration (HTI) and temperature-path
thermodynamic integration (TI) use `template_ff` instead of a Deep Potential
model:

- `in.mlip` contains the LAMMPS force-field commands;
- `input.nn`, `scaling.data`, and `weights.*.data` are the n2p2 model files;
- `water.lmp` is a 432-molecule liquid-water configuration;
- `hti_water.json` defines the three-stage liquid-water reference path;
- `ti.json` defines an NPT temperature path from 260 K to 320 K at 1 bar.

From this directory, generate the HTI anchor and the four TI tasks with:

```bash
dpti hti_water gen hti_water.json -o hti_water
dpti ti gen ti.json -o ti
```

During the MLIP-on stage, DPTI wraps the pair style from `in.mlip` in
`pair_style hybrid/scaled` with `v_LAMBDA`. This scales MLIP forces, energies,
and stresses while `compute pair hdnnp` reports the unscaled MLIP energy used
as the HTI integrand.

Each generated task contains links to the configuration and n2p2 files. To
submit the tasks, use a DPTI machine file whose LAMMPS command points to a build
that provides the `hdnnp` pair style:

```bash
dpti hti_water run hti_water /path/to/machine.json 00
dpti hti_water run hti_water /path/to/machine.json 01
dpti hti_water run hti_water /path/to/machine.json 02
dpti ti run ti /path/to/machine.json
```

The supplied trajectory length and statistical settings are representative of
the production calculation from which this example was adapted. Adjust them
when using the example for a shorter smoke test.
