#!/usr/bin/python3

import numpy as np


def _poscar_coordinate_records(lines):
    """Return header-derived element ownership for POSCAR coordinate lines."""
    names = lines[5].split()
    counts = [int(ii) for ii in lines[6].split()]
    if len(names) != len(counts):
        raise ValueError("POSCAR element names and counts must have equal lengths")

    coordinate_mode_index = 7
    if lines[coordinate_mode_index].strip().lower().startswith("s"):
        coordinate_mode_index += 1
    coordinate_start = coordinate_mode_index + 1
    natoms = sum(counts)
    positions = lines[coordinate_start : coordinate_start + natoms]
    if len(positions) != natoms:
        raise ValueError("POSCAR contains fewer coordinate lines than declared atoms")

    header_elements = [
        name for name, count in zip(names, counts) for _ in range(count)
    ]
    explicit_elements = [line.split()[-1] if line.split() else "" for line in positions]
    if all(element in names for element in explicit_elements):
        elements = explicit_elements
    else:
        # Standard POSCAR coordinates are unlabeled and follow header count order.
        elements = header_elements
    return names, counts, coordinate_start, list(zip(elements, positions))


def _write_grouped_poscar(poscar_in, poscar_out, ordered_names):
    with open(poscar_in) as fp:
        lines = fp.read().splitlines()
    names, counts, coordinate_start, records = _poscar_coordinate_records(lines)
    unique_names = list(dict.fromkeys(names))
    if len(ordered_names) != len(set(ordered_names)) or set(ordered_names) != set(
        unique_names
    ):
        raise ValueError("requested POSCAR order must contain each element exactly once")

    grouped = {
        name: [line for element, line in records if element == name]
        for name in ordered_names
    }
    new_counts = [len(grouped[name]) for name in ordered_names]
    coordinate_lines = [line for name in ordered_names for line in grouped[name]]
    ret = lines[:5]
    ret.append(" ".join(ordered_names))
    ret.append(" ".join(str(count) for count in new_counts))
    ret.extend(lines[7:coordinate_start])
    ret.extend(coordinate_lines)
    ret.extend(lines[coordinate_start + sum(counts) :])
    with open(poscar_out, "w") as fp:
        fp.write("\n".join(ret) + "\n")


def regulate_poscar(poscar_in, poscar_out):
    """Merge duplicate POSCAR element groups while retaining all coordinates."""
    with open(poscar_in) as fp:
        lines = fp.read().splitlines()
    names = lines[5].split()
    _write_grouped_poscar(poscar_in, poscar_out, list(dict.fromkeys(names)))


def sort_poscar(poscar_in, poscar_out, new_names):
    """Reorder POSCAR element groups using header counts for unlabeled coordinates."""
    _write_grouped_poscar(poscar_in, poscar_out, new_names)


def perturb_xz(poscar_in, poscar_out, pert=0.01):
    with open(poscar_in) as fp:
        lines = fp.read().split("\n")
    zz = lines[4]
    az = [float(ii) for ii in zz.split()]
    az[0] += pert
    zz = [str(ii) for ii in az]
    zz = " ".join(zz)
    lines[4] = zz
    with open(poscar_out, "w") as fp:
        fp.write("\n".join(lines))


def reciprocal_box(box):
    rbox = np.linalg.inv(box)
    rbox = rbox.T
    # rbox = rbox / np.linalg.det(box)
    # print(np.matmul(box, rbox.T))
    # print(rbox)
    return rbox


def _poscar_natoms(lines):
    numb_atoms = 0
    for ii in lines[6].split():
        numb_atoms += int(ii)
    return numb_atoms


def _poscar_scale_direct(str_in, scale):
    lines = str_in.copy()
    numb_atoms = _poscar_natoms(lines)
    pscale = float(lines[1])
    pscale = pscale * scale
    lines[1] = str(pscale) + "\n"
    return lines


def _poscar_scale_cartesian(str_in, scale):
    lines = str_in.copy()
    numb_atoms = _poscar_natoms(lines)
    # scale box
    for ii in range(2, 5):
        boxl = lines[ii].split()
        boxv = [float(ii) for ii in boxl]
        boxv = np.array(boxv) * scale
        lines[ii] = f"{boxv[0]:.16e} {boxv[1]:.16e} {boxv[2]:.16e}\n"
    # scale coord
    for ii in range(8, 8 + numb_atoms):
        cl = lines[ii].split()
        cv = [float(ii) for ii in cl]
        cv = np.array(cv) * scale
        lines[ii] = f"{cv[0]:.16e} {cv[1]:.16e} {cv[2]:.16e}\n"
    return lines


def poscar_natoms(poscar_in):
    with open(poscar_in) as fin:
        lines = list(fin)
    return _poscar_natoms(lines)


def poscar_scale(poscar_in, poscar_out, scale):
    with open(poscar_in) as fin:
        lines = list(fin)
    if "D" == lines[7][0] or "d" == lines[7][0]:
        lines = _poscar_scale_direct(lines, scale)
    elif "C" == lines[7][0] or "c" == lines[7][0]:
        lines = _poscar_scale_cartesian(lines, scale)
    else:
        raise RuntimeError(f"Unknow poscar coord style at line 7: {lines[7]}")
    with open(poscar_out, "w") as fout:
        fout.write("".join(lines))


def poscar_vol(poscar_in):
    with open(poscar_in) as fin:
        lines = list(fin)
    box = []
    for ii in range(2, 5):
        words = lines[ii].split()
        vec = [float(jj) for jj in words]
        box.append(vec)
    scale = float(lines[1].split()[0])
    box = np.array(box)
    box *= scale
    return np.linalg.det(box)
