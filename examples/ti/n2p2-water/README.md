# n2p2 water TI example

This example demonstrates temperature-path thermodynamic integration with a
Behler-Parrinello neural network potential through the LAMMPS `hdnnp` pair
style. It uses `template_ff` instead of a Deep Potential model:

- `in.mlip` contains the LAMMPS force-field commands;
- `input.nn`, `scaling.data`, and `weights.*.data` are the n2p2 model files;
- `water.lmp` is a 432-molecule liquid-water configuration;
- `ti.json` defines an NPT temperature path from 260 K to 320 K at 1 bar.

From this directory, generate the four TI tasks with:

```bash
dpti ti gen ti.json -o ti
```

Each generated task contains links to the configuration and n2p2 files, and its
`in.lammps` embeds the contents of `in.mlip`. To submit the tasks, use a DPTI
machine file whose LAMMPS command points to a build that provides the `hdnnp`
pair style:

```bash
dpti ti run ti /path/to/machine.json
```

The supplied trajectory length and statistical settings are representative of
the production calculation from which this example was adapted. Adjust them
when using the example for a shorter smoke test.
