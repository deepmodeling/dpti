<a href="./doc/logo.md"><picture><source media="(prefers-color-scheme: dark)" srcset="./doc/_static/logo.png"><source media="(prefers-color-scheme: light)" srcset="./doc/_static/logo.png"><img alt="dpti logo" src="./doc/_static/logo.png" width="620"></picture></a>

______________________________________________________________________

# dpti

`dpti` is a Python package for automating thermodynamic integration (TI)
workflows used to compute free energies and phase diagrams from molecular
dynamics simulations.

The package generates LAMMPS input files, submits or runs the corresponding MD
tasks through `dpdispatcher`, post-processes simulation outputs, evaluates
Helmholtz and Gibbs free energies, estimates statistical and numerical
integration errors, and propagates coexistence points into phase boundaries.

## Features

- Equilibration tasks in `NPT` and `NVT` ensembles.
- Hamiltonian thermodynamic integration (HTI) for atomic solids, atomic liquids,
  ice, and liquid water.
- Temperature and pressure thermodynamic integration (TTI and pTI) for
  propagating free energies along isobars and isotherms.
- Gibbs-Duhem integration (GDI) for propagating phase boundaries from
  coexistence points.
- Adaptive refinement of HTI and TI grids based on integration-error estimates.
- Local and HPC execution through `dpdispatcher`.

## Installation

Install the released package with `pip`:

```bash
pip install dpti
```

For development, install from a local checkout:

```bash
git clone https://github.com/deepmodeling/dpti.git
cd dpti
pip install -e .
```

Check the installation with:

```bash
dpti --help
```

`dpti` prepares and post-processes MD tasks. To run the generated simulations,
you also need a working LAMMPS executable and the force-field backend required by
your input files, such as DeePMD-kit for Deep Potential models.

## Basic Workflow

A typical phase-diagram calculation follows five stages.

1. Run an `NPT` equilibration to determine the average cell at a chosen
   pressure and temperature.
2. Run an `NVT` equilibration at that cell to prepare the initial configuration.
3. Run HTI to compute the absolute free energy at the anchor state.
4. Run TTI or pTI to propagate the free energy over a temperature or pressure
   range.
5. Run GDI from a coexistence point to propagate the phase boundary.

Most modules follow the same command pattern:

```bash
dpti <module> gen input.json -o job
dpti <module> run job machine.json
dpti <module> compute job
```

The `machine.json` file describes how tasks are executed by `dpdispatcher`. It
can point to a local machine, a Slurm/PBS cluster, or another supported backend.

## Command-Line Modules

| Module | Purpose |
| --- | --- |
| `equi` | Generate, run, and analyze equilibration simulations. |
| `hti` | HTI for atomic solids. |
| `hti_liq` | HTI for atomic liquids using soft-LJ intermediate states. |
| `hti_ice` | HTI for ice phases. |
| `hti_water` | HTI for liquid water. |
| `ti` | TTI or pTI for atomic systems. |
| `ti_water` | TTI or pTI for water systems. |
| `gdi` | Gibbs-Duhem integration of phase boundaries. |
| `mti` | Mass thermodynamic integration for quantum free-energy corrections. |

Use `-h` at any level to inspect the available options:

```bash
dpti -h
dpti hti -h
dpti hti gen -h
dpti ti compute -h
```

## Examples

Example input files are included in `examples/`. The snippets below show the
command structure; production calculations require physically meaningful input
JSON files, potential files, configurations, and machine settings.

### Equilibration

```bash
cd examples/equi
dpti equi gen npt.json -o npt
dpti equi run npt ../machine.json
dpti equi compute npt

dpti equi gen nvt.json -o nvt
dpti equi run nvt ../machine.json
dpti equi compute nvt
```

### HTI

For a one-step atomic-solid HTI path:

```bash
cd examples/hti
dpti hti gen hti.json -o hti
dpti hti run hti ../machine.json one-step
dpti hti compute hti --npt ../equi/npt
```

For a three-step path, use `-s three-step` and run the generated sub-tasks:

```bash
dpti hti gen hti.json -s three-step -o hti
dpti hti run hti ../machine.json 00
dpti hti run hti ../machine.json 01
dpti hti run hti ../machine.json 02
dpti hti compute hti --npt ../equi/npt
```

Atomic liquids use the `hti_liq` module; ice and liquid water use `hti_ice` and
`hti_water`, respectively.

### TTI and pTI

The `ti` module generates and computes temperature or pressure integration tasks
from a JSON input file. The path type is specified in the JSON file.

```bash
cd examples/ti
dpti ti gen ti.t.json -o ti_t
dpti ti run ti_t ../machine.json
dpti ti compute ti_t -H ../hti/hti

dpti ti gen ti.p.json -o ti_p
dpti ti run ti_p ../machine.json
dpti ti compute ti_p -H ../hti/hti
```

The `-H/--hti` option reads the anchor free energy from an HTI job directory.
Alternatively, provide the starting free energy explicitly with `-e/--Eo`,
`-E/--Eo-err`, and `-t/--To`.

The self-contained
[`examples/ti/n2p2-water`](examples/ti/n2p2-water/README.md) case shows how to
use a LAMMPS force-field template and its supporting n2p2 model files instead of
a Deep Potential model.

### GDI

GDI propagates a phase boundary from an initial coexistence point. The phase
settings are read from the main JSON file, while the integration direction,
initial value, tolerances, and output path can be provided either through a
GDI-data JSON file or through command-line options.

```bash
cd examples/gdi
dpti gdi pb.json machine.json -g gdidata.json
```

## Input Files

`dpti` is controlled by JSON files. The exact fields depend on the module and
system type, but the inputs usually describe:

- the initial configuration and potential files;
- thermodynamic conditions and integration grids;
- LAMMPS settings such as timestep, number of steps, thermostat/barostat
  parameters, and output frequency;
- execution settings through `machine.json`.

Start from the files in `examples/` and adapt them to the target system.

## Legacy Airflow Support

The recommended interface is the command-line workflow described above. The
repository still contains legacy Airflow DAG helpers under `dpti/dags/` for
users who need Airflow-based orchestration, but new users should start with the
CLI.

## License

`dpti` is distributed under the GNU Lesser General Public License v3.0. See
`LICENSE` for details.
