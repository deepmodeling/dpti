"""Fit a soft-core Lennard-Jones reference to DeepMD-format data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    """DeepMD-format coordinates, boxes, energies, forces, and atom types."""

    coord: np.ndarray
    box: np.ndarray
    energy: np.ndarray
    force: np.ndarray
    atype: np.ndarray
    type_map: list[str]


def pair_index(type_i: np.ndarray, type_j: np.ndarray, ntypes: int) -> np.ndarray:
    """Return packed upper-triangular pair indices for atom-type pairs."""
    lo = np.minimum(type_i, type_j)
    hi = np.maximum(type_i, type_j)
    return lo * ntypes - lo * (lo - 1) // 2 + hi - lo


def pair_labels(type_map: list[str]) -> list[str]:
    """Return pair labels in packed upper-triangular order."""
    return [
        f"{type_map[i]}-{type_map[j]}"
        for i in range(len(type_map))
        for j in range(i, len(type_map))
    ]


def pair_lists(
    atype: np.ndarray, ntypes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate unique atom pairs and their parameter indices."""
    ii, jj = np.triu_indices(atype.size, k=1)
    pp = pair_index(atype[ii], atype[jj], ntypes)
    return ii.astype(np.int32), jj.astype(np.int32), pp.astype(np.int32)


def load_deepmd(
    path: Path,
    stride: int = 1,
    skip_frames: int = 0,
    max_frames: int | None = None,
) -> Dataset:
    """Load all ``set.*`` directories of a DeepMD NumPy dataset."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if skip_frames < 0:
        raise ValueError("skip-frames must be non-negative")
    set_dirs = sorted(p for p in path.glob("set.*") if p.is_dir())
    if not set_dirs:
        raise ValueError(f"No set.* directories found in {path}")

    atype = np.atleast_1d(np.loadtxt(path / "type.raw", dtype=np.int64))
    type_map = (path / "type_map.raw").read_text().split()
    if not type_map:
        raise ValueError("type_map.raw is empty")
    if np.any(atype < 0) or np.any(atype >= len(type_map)):
        raise ValueError("type.raw contains an index absent from type_map.raw")
    if not np.array_equal(np.unique(atype), np.arange(len(type_map))):
        raise ValueError("Every entry in type_map.raw must occur in type.raw")

    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("coord", "box", "energy", "force")
    }
    for set_dir in set_dirs:
        for key in arrays:
            arrays[key].append(np.load(set_dir / f"{key}.npy"))
    loaded = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}

    frame_slice = slice(skip_frames, None, stride)
    loaded = {key: value[frame_slice] for key, value in loaded.items()}
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max-frames must be positive")
        loaded = {key: value[:max_frames] for key, value in loaded.items()}
    nframes = loaded["coord"].shape[0]
    if nframes == 0:
        raise ValueError("Frame selection leaves no data")
    if any(value.shape[0] != nframes for value in loaded.values()):
        raise ValueError("DeepMD arrays contain different numbers of frames")

    natoms = atype.size
    return Dataset(
        coord=loaded["coord"].reshape(nframes, natoms, 3),
        box=loaded["box"].reshape(nframes, 3, 3),
        energy=loaded["energy"].reshape(nframes),
        force=loaded["force"].reshape(nframes, natoms, 3),
        atype=atype,
        type_map=type_map,
    )


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply the minimum-image convention for a general periodic cell."""
    fractional = displacement @ np.linalg.inv(box)
    fractional -= np.rint(fractional)
    return fractional @ box


def predict_one_frame(
    coord: np.ndarray,
    box: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    pp: np.ndarray,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    activation: np.ndarray,
    n_exp: float,
    alpha_lj: float,
    cutoff: float,
) -> tuple[float, np.ndarray]:
    """Evaluate LAMMPS ``lj/cut/soft`` energy and forces for one frame."""
    displacement = minimum_image(coord[ii] - coord[jj], box)
    r2 = np.einsum("ij,ij->i", displacement, displacement)
    mask = (r2 > 0.0) & (r2 < cutoff**2)
    force = np.zeros_like(coord)
    if not np.any(mask):
        return 0.0, force

    displacement = displacement[mask]
    r = np.sqrt(r2[mask])
    pair_type = pp[mask]
    eps = epsilon[pair_type]
    sig = sigma[pair_type]
    act = activation[pair_type]
    denominator = alpha_lj * (1.0 - act) ** 2 + (r / sig) ** 6
    pair_energy = act**n_exp * 4.0 * eps * (denominator**-2 - denominator**-1)

    ddenominator_dr = 6.0 * r**5 / sig**6
    denergy_dr = (
        act**n_exp
        * 4.0
        * eps
        * (-2.0 * denominator**-3 + denominator**-2)
        * ddenominator_dr
    )
    pair_force = -(denergy_dr / r)[:, None] * displacement
    np.add.at(force, ii[mask], pair_force)
    np.add.at(force, jj[mask], -pair_force)
    return float(np.sum(pair_energy)), force


def global_rmse(
    predicted_energy: np.ndarray,
    target_energy: np.ndarray,
    predicted_force: np.ndarray,
    target_force: np.ndarray,
    natoms: int,
) -> dict[str, float]:
    """Compute globally centered energy and force RMSE values."""
    energy_residual = predicted_energy - target_energy
    energy_residual -= np.mean(energy_residual)
    energy_rmse = float(np.sqrt(np.mean(energy_residual**2)))
    return {
        "e_rmse": energy_rmse,
        "e_rmse_atom": energy_rmse / natoms,
        "f_rmse": float(np.sqrt(np.mean((predicted_force - target_force) ** 2))),
    }


def _expand(values: list[float], npairs: int, name: str) -> np.ndarray:
    if len(values) == 1:
        return np.full(npairs, values[0], dtype=float)
    if len(values) != npairs:
        raise ValueError(
            f"{name} requires either one value or {npairs} pair-specific values"
        )
    return np.asarray(values, dtype=float)


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-8, 1.0 - 1e-8)
    return np.log(values / (1.0 - values))


def fit(args: argparse.Namespace) -> None:
    """Fit parameters with minibatch Adam optimization using JAX."""
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:
        raise ImportError(
            "Soft-LJ fitting requires JAX and Optax; install dpti[soft-lj]."
        ) from exc

    jax.config.update("jax_enable_x64", args.x64)
    data = load_deepmd(args.data, args.stride, args.skip_frames, args.max_frames)
    ntypes = len(data.type_map)
    npairs = ntypes * (ntypes + 1) // 2
    ii_np, jj_np, pp_np = pair_lists(data.atype, ntypes)
    labels = pair_labels(data.type_map)
    nframes, natoms = data.coord.shape[:2]

    if args.cutoff <= 0 or args.n_exp <= 0 or args.alpha_lj < 0:
        raise ValueError(
            "cutoff and n-exp must be positive and alpha-lj must be non-negative"
        )
    if min(args.steps, args.batch_size, args.eval_batch_size, args.eval_every) < 1:
        raise ValueError(
            "steps, batch-size, eval-batch-size, and eval-every must be positive"
        )
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("grad-clip must be positive")
    if args.epsilon_max <= 0 or args.sigma_max <= 0:
        raise ValueError("epsilon-max and sigma-max must be positive")

    init_epsilon = _expand(args.init_epsilon, npairs, "init-epsilon")
    init_sigma = _expand(args.init_sigma, npairs, "init-sigma")
    activation_np = _expand(args.activation, npairs, "activation")
    if np.any(init_epsilon <= 0) or np.any(init_epsilon >= args.epsilon_max):
        raise ValueError("init-epsilon values must lie between zero and epsilon-max")
    if np.any(init_sigma <= 0) or np.any(init_sigma >= args.sigma_max):
        raise ValueError("init-sigma values must lie between zero and sigma-max")
    if np.any(activation_np <= 0) or np.any(activation_np >= 1):
        raise ValueError("activation values must lie strictly between zero and one")

    raw_parts = [
        _logit(init_epsilon / args.epsilon_max),
        _logit(init_sigma / args.sigma_max),
    ]
    if args.fit_activation:
        raw_parts.append(_logit(activation_np))
    dtype = jnp.float64 if args.x64 else jnp.float32
    theta = jnp.asarray(np.concatenate(raw_parts), dtype=dtype)
    activation_fixed = jnp.asarray(activation_np, dtype=dtype)
    epsilon_max = jnp.asarray(args.epsilon_max, dtype=dtype)
    sigma_max = jnp.asarray(args.sigma_max, dtype=dtype)

    coord = jnp.asarray(data.coord, dtype=dtype)
    box = jnp.asarray(data.box, dtype=dtype)
    energy = jnp.asarray(data.energy, dtype=dtype)
    force = jnp.asarray(data.force, dtype=dtype)
    ii = jnp.asarray(ii_np)
    jj = jnp.asarray(jj_np)
    pp = jnp.asarray(pp_np)
    cutoff2 = jnp.asarray(args.cutoff**2, dtype=dtype)

    def unpack(theta_j):
        epsilon = epsilon_max * jax.nn.sigmoid(theta_j[:npairs])
        sigma = sigma_max * jax.nn.sigmoid(theta_j[npairs : 2 * npairs])
        if args.fit_activation:
            activation = jax.nn.sigmoid(theta_j[2 * npairs :])
        else:
            activation = activation_fixed
        return epsilon, sigma, activation

    def predict_frame(theta_j, coord_f, box_f):
        epsilon, sigma, activation = unpack(theta_j)
        displacement = coord_f[ii] - coord_f[jj]
        fractional = displacement @ jnp.linalg.inv(box_f)
        displacement = (fractional - jnp.rint(fractional)) @ box_f
        r2 = jnp.sum(displacement**2, axis=1)
        mask = (r2 > 0.0) & (r2 < cutoff2)
        r = jnp.sqrt(jnp.where(mask, r2, 1.0))
        eps = epsilon[pp]
        sig = sigma[pp]
        act = activation[pp]
        denominator = args.alpha_lj * (1.0 - act) ** 2 + (r / sig) ** 6
        pair_energy = act**args.n_exp * 4.0 * eps * (denominator**-2 - denominator**-1)
        pair_energy = jnp.where(mask, pair_energy, 0.0)
        ddenominator_dr = 6.0 * r**5 / sig**6
        denergy_dr = (
            act**args.n_exp
            * 4.0
            * eps
            * (-2.0 * denominator**-3 + denominator**-2)
            * ddenominator_dr
        )
        pair_force = -(denergy_dr / r)[:, None] * displacement
        pair_force = jnp.where(mask[:, None], pair_force, 0.0)
        predicted_force = jnp.zeros_like(coord_f)
        predicted_force = predicted_force.at[ii].add(pair_force)
        predicted_force = predicted_force.at[jj].add(-pair_force)
        return jnp.sum(pair_energy), predicted_force

    predict_batch = jax.jit(jax.vmap(predict_frame, in_axes=(None, 0, 0)))

    if args.decay_steps is None:
        learning_rate = args.learning_rate
    else:
        if args.decay_steps < 1 or not 0 < args.decay_rate <= 1:
            raise ValueError(
                "decay-steps must be positive and decay-rate must lie in (0, 1]"
            )
        learning_rate = optax.exponential_decay(
            args.learning_rate, args.decay_steps, args.decay_rate
        )
    adam = optax.adam(learning_rate)
    optimizer = (
        adam
        if args.grad_clip is None
        else optax.chain(optax.clip_by_global_norm(args.grad_clip), adam)
    )
    state = optimizer.init(theta)

    @jax.jit
    def train_step(
        theta_j, state_j, coord_b, box_b, energy_b, force_b, force_weight, energy_weight
    ):
        def weighted_loss(parameters):
            predicted_energy, predicted_force = jax.vmap(
                predict_frame, in_axes=(None, 0, 0)
            )(parameters, coord_b, box_b)
            force_mse = jnp.mean((predicted_force - force_b) ** 2)
            energy_residual = predicted_energy - energy_b
            energy_residual -= jnp.mean(energy_residual)
            energy_mse_atom = jnp.mean(energy_residual**2) / natoms**2
            return force_weight * force_mse + energy_weight * energy_mse_atom

        loss, gradient = jax.value_and_grad(weighted_loss)(theta_j)
        updates, state_j = optimizer.update(gradient, state_j, theta_j)
        return optax.apply_updates(theta_j, updates), state_j, loss

    def evaluate(theta_j):
        energy_chunks = []
        force_chunks = []
        for start in range(0, nframes, args.eval_batch_size):
            stop = min(start + args.eval_batch_size, nframes)
            predicted_energy, predicted_force = predict_batch(
                theta_j, coord[start:stop], box[start:stop]
            )
            energy_chunks.append(np.asarray(jax.device_get(predicted_energy)))
            force_chunks.append(np.asarray(jax.device_get(predicted_force)))
        return global_rmse(
            np.concatenate(energy_chunks),
            data.energy,
            np.concatenate(force_chunks),
            data.force,
            natoms,
        )

    rng = np.random.default_rng(args.seed)
    history_step: list[int] = []
    history_loss: list[float] = []
    history_e_rmse_atom: list[float] = []
    history_f_rmse: list[float] = []
    best_score = np.inf
    best_step = 0
    best_theta = theta
    best_metrics: dict[str, float] = {}
    force_weight_start = (
        args.force_weight
        if args.start_force_weight is None
        else args.start_force_weight
    )
    force_weight_limit = (
        args.force_weight
        if args.limit_force_weight is None
        else args.limit_force_weight
    )
    energy_weight_start = (
        args.energy_weight
        if args.start_energy_weight is None
        else args.start_energy_weight
    )
    energy_weight_limit = (
        args.energy_weight
        if args.limit_energy_weight is None
        else args.limit_energy_weight
    )
    if (
        min(
            force_weight_start,
            force_weight_limit,
            energy_weight_start,
            energy_weight_limit,
        )
        < 0
    ):
        raise ValueError("loss weights must be non-negative")

    def scheduled_weight(step: int, start: float, limit: float) -> float:
        if args.decay_steps is None:
            fraction = max(0.0, 1.0 - step / args.steps)
        else:
            fraction = args.decay_rate ** (step / args.decay_steps)
        return limit + (start - limit) * fraction

    print(
        f"Loaded {nframes} frames with {natoms} atoms and types {data.type_map} from {args.data}"
    )
    print(f"Fitting {', '.join(labels)} with batch_size={args.batch_size}")
    for step in range(1, args.steps + 1):
        force_weight = scheduled_weight(step, force_weight_start, force_weight_limit)
        energy_weight = scheduled_weight(step, energy_weight_start, energy_weight_limit)
        size = min(args.batch_size, nframes)
        indices = rng.choice(nframes, size=size, replace=False)
        theta, state, batch_loss = train_step(
            theta,
            state,
            coord[indices],
            box[indices],
            energy[indices],
            force[indices],
            jnp.asarray(force_weight, dtype=dtype),
            jnp.asarray(energy_weight, dtype=dtype),
        )
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(theta)
            score = metrics[args.select_metric]
            history_step.append(step)
            history_loss.append(float(jax.device_get(batch_loss)))
            history_e_rmse_atom.append(metrics["e_rmse_atom"])
            history_f_rmse.append(metrics["f_rmse"])
            print(
                f"step {step:6d} batch_loss={history_loss[-1]:.8g} "
                f"weights(E,F)=({energy_weight:.4g},{force_weight:.4g}) "
                f"global_E_RMSE/atom={metrics['e_rmse_atom']:.6g} eV "
                f"global_F_RMSE={metrics['f_rmse']:.6g} eV/A",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_step = step
                best_theta = theta
                best_metrics = metrics

    epsilon, sigma, activation = (
        np.asarray(jax.device_get(value)) for value in unpack(best_theta)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fitted soft-core Lennard-Jones parameters for LAMMPS",
        f"# source: {args.data}",
        f"# global centered energy RMSE per atom: {best_metrics['e_rmse_atom']:.8g} eV",
        f"# global force RMSE: {best_metrics['f_rmse']:.8g} eV/A",
        f"pair_style lj/cut/soft {args.n_exp:.16g} {args.alpha_lj:.16g} {args.cutoff:.16g}",
    ]
    coefficient = 0
    for i in range(ntypes):
        for j in range(i, ntypes):
            lines.append(
                f"pair_coeff {i + 1} {j + 1} {epsilon[coefficient]:.16g} "
                f"{sigma[coefficient]:.16g} {activation[coefficient]:.16g}  "
                f"# {labels[coefficient]}"
            )
            coefficient += 1
    args.output.write_text("\n".join(lines) + "\n")
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.history_output,
        step=np.asarray(history_step),
        batch_loss=np.asarray(history_loss),
        global_e_rmse_atom=np.asarray(history_e_rmse_atom),
        global_f_rmse=np.asarray(history_f_rmse),
        epsilon=epsilon,
        sigma=sigma,
        activation=activation,
        best_step=np.asarray(best_step),
    )
    print(f"Wrote parameters from step {best_step} to {args.output}")
    print(f"Wrote optimization history to {args.history_output}")


def add_module_subparsers(main_subparsers) -> None:
    """Add the ``soft_lj fit`` command to the DPTI parser."""
    module_parser = main_subparsers.add_parser(
        "soft_lj", help="fit a soft-core Lennard-Jones reference system"
    )
    subparsers = module_parser.add_subparsers(dest="command", required=True)
    parser = subparsers.add_parser("fit", help="fit soft-LJ parameters to DeepMD data")
    parser.add_argument("DATA", type=Path, help="DeepMD NumPy dataset directory")
    parser.add_argument("-o", "--output", type=Path, default=Path("soft_lj.params"))
    parser.add_argument(
        "--history-output", type=Path, default=Path("soft_lj_history.npz")
    )
    parser.add_argument("--cutoff", type=float, default=7.5, help="cutoff in angstrom")
    parser.add_argument(
        "--n-exp", type=float, default=2.0, help="soft-LJ activation exponent"
    )
    parser.add_argument(
        "--alpha-lj", type=float, default=0.5, help="soft-core coefficient"
    )
    parser.add_argument("--activation", nargs="+", type=float, default=[0.5])
    parser.add_argument(
        "--fit-activation",
        action="store_true",
        help="optimize activation instead of keeping --activation fixed",
    )
    parser.add_argument("--init-epsilon", nargs="+", type=float, default=[0.01])
    parser.add_argument("--init-sigma", nargs="+", type=float, default=[2.0])
    parser.add_argument("--epsilon-max", type=float, default=10.0)
    parser.add_argument("--sigma-max", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--decay-steps", type=int)
    parser.add_argument("--decay-rate", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--start-force-weight", type=float)
    parser.add_argument("--limit-force-weight", type=float)
    parser.add_argument("--start-energy-weight", type=float)
    parser.add_argument("--limit-energy-weight", type=float)
    parser.add_argument(
        "--select-metric", choices=["f_rmse", "e_rmse_atom"], default="f_rmse"
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip-frames", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument(
        "--x64", action="store_true", help="enable JAX 64-bit precision"
    )
    parser.set_defaults(func=_handle_fit)


def _handle_fit(args: argparse.Namespace) -> None:
    args.data = args.DATA
    fit(args)
