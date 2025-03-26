# %%
import base64
import io
import json
import os
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %%

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from typing import Optional

from dpti.workflows.simulations.ti_sim import plot_ti_result


# %%
def plot_free_energy_comparison(
    phase1_result: dict,
    phase2_result: dict,
    phase1_label: str = "Phase 1",
    phase2_label: str = "Phase 2",
) -> str:
    """Compare free energy curves of two phases and find their intersection point.

    Args:
        phase1_result: TI simulation results for phase 1
        phase2_result: TI simulation results for phase 2
        phase1_label: Label for phase 1 in the plot
        phase2_label: Label for phase 2 in the plot

    Returns
    -------
    base64 encoded string of the plot image
    """
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 14})

    # Extract data for both phases
    p1_temps = np.array(phase1_result["data"]["all_temps"])
    p1_fe = np.array(phase1_result["data"]["all_fe"])
    p1_tot_err = np.sqrt(
        np.array(phase1_result["data"]["all_fe_stat_err"]) ** 2
        + np.array(phase1_result["data"]["all_fe_inte_err"]) ** 2
    )

    p2_temps = np.array(phase2_result["data"]["all_temps"])
    p2_fe = np.array(phase2_result["data"]["all_fe"])
    p2_tot_err = np.sqrt(
        np.array(phase2_result["data"]["all_fe_stat_err"]) ** 2
        + np.array(phase2_result["data"]["all_fe_inte_err"]) ** 2
    )

    # Plot phase 1
    plt.errorbar(
        p1_temps,
        p1_fe,
        yerr=p1_tot_err,
        fmt="o-",
        label=phase1_label,
        color="blue",
        capsize=5,
        ecolor="lightblue",
        elinewidth=2,
    )

    # Plot phase 2
    plt.errorbar(
        p2_temps,
        p2_fe,
        yerr=p2_tot_err,
        fmt="o-",
        label=phase2_label,
        color="red",
        capsize=5,
        ecolor="lightcoral",
        elinewidth=2,
    )

    # Find intersection using linear interpolation
    from scipy import interpolate

    # Create interpolation functions for both curves
    f_phase1 = interpolate.interp1d(p1_temps, p1_fe, kind="linear")
    f_phase2 = interpolate.interp1d(p2_temps, p2_fe, kind="linear")

    # Find intersection in overlapping temperature range
    t_min = max(p1_temps.min(), p2_temps.min())
    t_max = min(p1_temps.max(), p2_temps.max())
    t_test = np.linspace(t_min, t_max, 1000)

    try:
        # Calculate free energies at test points
        fe_phase1 = f_phase1(t_test)
        fe_phase2 = f_phase2(t_test)
        diff = fe_phase1 - fe_phase2
        cross_idx = np.where(np.diff(np.signbit(diff)))[0]

        if len(cross_idx) > 0:
            # Mark intersection point if found
            t_cross = t_test[cross_idx[0]]
            fe_cross = f_phase1(t_cross)

            plt.plot(t_cross, fe_cross, "go", markersize=10, label="Transition point")
            plt.annotate(
                f"T = {t_cross:.1f}K\nG = {fe_cross:.6f}eV",
                xy=(t_cross, fe_cross),
                xytext=(20, 20),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.5", "fc": "yellow", "alpha": 0.5},
                arrowprops={"arrowstyle": "->"},
            )
    except:
        print("No intersection point found in the overlapping temperature range")

    plt.xlabel("Temperature (K)")
    plt.ylabel("Gibbs Free Energy (eV/atom)")
    plt.title("Free Energy Comparison Between Phases")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Adjust y-axis range for better visualization
    all_fe = np.concatenate([p1_fe, p2_fe])
    y_range = np.max(all_fe) - np.min(all_fe)
    plt.ylim(np.min(all_fe) - y_range * 0.1, np.max(all_fe) + y_range * 0.1)

    plt.tight_layout()
    plt.show(block=False)

    # Save and encode the plot
    with BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return img_base64


# %%

with open(
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run3/TI_sim/new_job/result.json"
) as f:
    solid_result_info = json.load(f)

with open(
    "../examples/Sn_beta_quicktest/TI_liquid128_path_t_1000K_50000bar_run0/TI_sim/new_job/result.json"
) as f:
    liquid_result_info = json.load(f)

solid_img_base64 = plot_ti_result(result_info=solid_result_info)
liquid_img_base64 = plot_ti_result(result_info=liquid_result_info)

comparison_img_base64 = plot_free_energy_comparison(
    phase1_result=solid_result_info,
    phase2_result=liquid_result_info,
    phase1_label="Solid phase",
    phase2_label="Liquid phase",
)
# %%


# %%


def analyze_align_different_temp_starting_points(result_info: dict) -> str:
    """Analyze and align different temperature starting points in a TI simulation.

    Args:
        result_info: Dictionary containing TI simulation results

    Returns
    -------
    base64 encoded string of the plot image
    """
    pass


# %%


def compare_multiple_runs(result_files: list, labels: Optional[list] = None) -> str:
    """Compare multiple TI simulation runs with different random seeds.

    Args:
        result_files: List of paths to result.json files
        labels: Optional list of labels for each run

    Returns
    -------
    base64 encoded string of the plot image
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), height_ratios=[1, 1])
    plt.rcParams.update({"font.size": 14})

    # Load and process data
    all_temps = []
    all_fe = []
    all_errors = []

    for file_path in result_files:
        with open(file_path) as f:
            result = json.load(f)
            temps = np.array(result["data"]["all_temps"])
            fe = np.array(result["data"]["all_fe"])
            tot_err = np.sqrt(
                np.array(result["data"]["all_fe_stat_err"]) ** 2
                + np.array(result["data"]["all_fe_inte_err"]) ** 2
            )
            all_temps.append(temps)
            all_fe.append(fe)
            all_errors.append(tot_err)

    all_temps = np.array(all_temps)
    all_fe = np.array(all_fe)
    all_errors = np.array(all_errors)

    # Calculate statistics
    mean_fe = np.mean(all_fe, axis=0)
    std_fe = np.std(all_fe, axis=0)
    mean_temp = np.mean(all_temps, axis=0)

    # Upper subplot: Absolute values
    ax1.plot(mean_temp, mean_fe, "b-", label="Mean", linewidth=2)
    ax1.fill_between(
        mean_temp,
        mean_fe - std_fe,
        mean_fe + std_fe,
        alpha=0.2,
        color="blue",
        label="Standard deviation",
    )

    colors = plt.cm.rainbow(np.linspace(0, 1, len(result_files)))
    for i, (temps, fe) in enumerate(zip(all_temps, all_fe)):
        label = labels[i] if labels else f"Run {i+1}"
        ax1.plot(temps, fe, "--", color=colors[i], alpha=0.5, label=label)

    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Gibbs Free Energy (eV/atom)")
    ax1.set_title("Absolute Free Energy Values")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Lower subplot: Differences from mean
    for i, (temps, fe) in enumerate(zip(all_temps, all_fe)):
        label = labels[i] if labels else f"Run {i+1}"
        diff = fe - mean_fe
        ax2.plot(temps, diff, "-", color=colors[i], label=label)

    # Add zero line for reference
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=1, alpha=0.5)

    # Add standard deviation bands
    ax2.fill_between(
        mean_temp, -std_fe, std_fe, alpha=0.2, color="gray", label="Standard deviation"
    )

    ax2.set_xlabel("Temperature (K)")
    ax2.set_ylabel("Δ Free Energy (eV/atom)")
    ax2.set_title("Deviation from Mean Value")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show(block=False)
    # Save and encode the plot
    with BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return img_base64


# %%

result_files = [
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run0/TI_sim/new_job/result.json",
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run1/TI_sim/new_job/result.json",
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run2/TI_sim/new_job/result.json",
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_400K_50000bar_run0/TI_sim/new_job/result.json",
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_400K_50000bar_run1/TI_sim/new_job/result.json",
    "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_400K_50000bar_run2/TI_sim/new_job/result.json",
    # Add more result files as needed
]

# labels = [Path(result_files[ii]).stem.split('/')[-1] for ii in range(len(result_files))]
labels = [f'run{f.split("run")[1].split("/")[0]}' for f in result_files]
# labels = ['run0', 'run1', 'run2']  # Optional labels
comparison_img = compare_multiple_runs(result_files, labels)
# %%
import glob

result_files = glob.glob(
    "../examples/Sn_beta_quicktest/TI_liquid128_path_t_1000K_50000bar_run*/TI_sim/new_job/result.json"
)


result_files.extend(
    glob.glob(
        "../examples/Sn_beta_quicktest/TI_liquid128_path_t_1200K_50000bar_run*/TI_sim/new_job/result.json"
    )
)
result_files.extend(
    glob.glob(
        "../examples/Sn_beta_quicktest/TI_liquid128_path_t_920K_50000bar_run*/TI_sim/new_job/result.json"
    )
)
result_files.extend(
    glob.glob(
        "../examples/Sn_beta_quicktest/TI_liquid128_path_t_1080K_50000bar_run*/TI_sim/new_job/result.json"
    )
)


labels = [f'run{f.split("run")[1].split("/")[0]}' for f in result_files]

print(result_files)
# labels = ['run0', 'run1', 'run2']  # Optional labels
comparison_img = compare_multiple_runs(result_files, labels)


# %%

from typing import Dict, List


def plot_multiple_phase_comparison(
    phase1_results: List[Dict],
    phase2_results: List[Dict],
    phase1_label: str = "Phase 1",
    phase2_label: str = "Phase 2",
) -> Dict[str, str]:
    """Compare multiple results between two phases and generate statistical plots.

    Args:
        phase1_results: List of result dictionaries for phase 1
        phase2_results: List of result dictionaries for phase 2
        phase1_label: Label for phase 1 in plots
        phase2_label: Label for phase 2 in plots

    Returns
    -------
    Dict containing base64 encoded plot images:
    - 'all_comparisons': Plot showing all N*M comparisons
    - 'statistical_distribution': Plot showing distribution and mean
    - 'mean_comparison': Plot showing mean values with error bars
    """
    import base64
    from itertools import product

    import matplotlib.pyplot as plt
    import numpy as np

    # 存储所有比较的结果
    all_comparisons = []
    temps = phase1_results[0]["data"]["all_temps"]  # 假设温度点相同

    # 生成所有可能的组合比较
    for p1_result, p2_result in product(phase1_results, phase2_results):
        fe_diff = np.array(p2_result["data"]["all_fe"]) - np.array(
            p1_result["data"]["all_fe"]
        )
        all_comparisons.append(fe_diff)

    all_comparisons = np.array(all_comparisons)
    mean_diff = np.mean(all_comparisons, axis=0)
    std_diff = np.std(all_comparisons, axis=0)

    # 1. 所有比较的叠加图
    plt.figure(figsize=(10, 6))
    for i, diff in enumerate(all_comparisons):
        plt.plot(temps, diff, alpha=0.3, label=f"Comparison {i+1}" if i == 0 else None)
    plt.plot(temps, mean_diff, "k-", linewidth=2, label="Mean")
    plt.fill_between(
        temps, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color="gray"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Free Energy Difference (eV)")
    plt.title(
        f"{phase2_label} - {phase1_label} Free Energy Difference\nAll Comparisons"
    )
    plt.legend()
    plt.grid(True)

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    all_comparisons_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 2. 分布图（在每个温度点）
    plt.figure(figsize=(12, 6))
    plt.violinplot(all_comparisons, positions=temps)
    plt.plot(temps, mean_diff, "r-", linewidth=2, label="Mean")
    plt.fill_between(
        temps, mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color="red"
    )
    plt.xlabel("Temperature (K)")
    plt.ylabel("Free Energy Difference (eV)")
    plt.title(f"{phase2_label} - {phase1_label} Free Energy Difference\nDistribution")
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    distribution_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 3. 平均值比较图（带误差棒）
    plt.figure(figsize=(10, 6))
    plt.errorbar(temps, mean_diff, yerr=std_diff, capsize=5)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Free Energy Difference (eV)")
    plt.title(
        f"{phase2_label} - {phase1_label} Free Energy Difference\nMean with Error Bars"
    )
    plt.grid(True)

    buf = io.BytesIO()
    plt.show(block=False)
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    mean_comparison_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # 添加: 计算所有交点温度
    def find_crossing_temp(fe_diff, temps):
        """找到自由能差为0的温度点（相变点）."""
        from scipy.interpolate import interp1d
        from scipy.optimize import root_scalar

        # 使用三次样条插值
        fe_interp = interp1d(temps, fe_diff, kind="cubic")

        try:
            # 在温度范围内寻找交点
            result = root_scalar(
                fe_interp, bracket=[temps[0], temps[-1]], method="brentq"
            )
            return result.root
        except ValueError:
            return None  # 如果在范围内没有交点

    # 收集所有交点温度
    crossing_temps = []
    for diff in all_comparisons:
        temp = find_crossing_temp(diff, temps)
        if temp is not None:
            crossing_temps.append(temp)

    # 热力图显示交点温度
    plt.figure(figsize=(10, 8))
    crossing_temps_matrix = np.array(crossing_temps).reshape(
        len(phase2_results), len(phase1_results)
    )
    mean_temp = np.mean(crossing_temps)
    std_temp = np.std(crossing_temps)

    # 使用对称的颜色映射，以平均值为中心
    vmin = mean_temp - 3 * std_temp
    vmax = mean_temp + 3 * std_temp

    sns.heatmap(
        crossing_temps_matrix,
        annot=True,  # 显示具体数值
        fmt=".1f",  # 数值格式保留一位小数
        cmap="RdYlBu_r",  # 使用红-黄-蓝配色，_r表示反转（红色表示高温）
        center=mean_temp,  # 将颜色中心设为平均值
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"Run {i+1}" for i in range(len(phase1_results))],
        yticklabels=[f"Run {i+1}" for i in range(len(phase2_results))],
        cbar_kws={"label": "Crossing Temperature (K)"},
    )

    plt.xlabel(f"{phase1_label} Runs")
    plt.ylabel(f"{phase2_label} Runs")
    plt.title(
        f"Phase Transition Temperatures (K)\nMean: {mean_temp:.1f}K ± {std_temp:.1f}K"
    )

    # 添加网格线使单元格更清晰
    plt.grid(False)

    buf = io.BytesIO()
    plt.show(block=False)
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "all_comparisons": all_comparisons_base64,
        "statistical_distribution": distribution_base64,
        "mean_comparison": mean_comparison_base64,
        "crossing_temps_heatmap": heatmap_base64,
        "numerical_data": {
            "temperatures": temps,
            "mean_difference": mean_diff.tolist(),
            "std_difference": std_diff.tolist(),
            "all_differences": all_comparisons.tolist(),
            "crossing_temperatures": {
                "values": crossing_temps,
                "mean": float(mean_temp),
                "std": float(std_temp),
                "matrix": crossing_temps_matrix.tolist(),
            },
        },
    }


# %%

# 加载多个结果
phase1_results = []
phase2_results = []

# 加载solid phase结果
for run in [0, 1, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase1_results.append(json.load(f))

for run in [0, 1, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_beta144xy_path_t_400K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase1_results.append(json.load(f))

for run in [0, 1, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_beta144xy_path_t_480K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase1_results.append(json.load(f))


# 加载liquid phase结果
for run in [0, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_liquid128_path_t_1000K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase2_results.append(json.load(f))

for run in [0, 1]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_liquid128_path_t_1200K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase2_results.append(json.load(f))

for run in [0, 1, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_liquid128_path_t_920K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase2_results.append(json.load(f))

for run in [0, 1, 2]:
    with open(
        f"../examples/Sn_beta_quicktest/TI_liquid128_path_t_1080K_50000bar_run{run}/TI_sim/new_job/result.json"
    ) as f:
        phase2_results.append(json.load(f))


# 生成比较图
comparison_results = plot_multiple_phase_comparison(
    phase1_results=phase1_results,
    phase2_results=phase2_results,
    phase1_label="Solid phase",
    phase2_label="Liquid phase",
)

# 可以将结果保存或展示
for plot_name, base64_data in comparison_results.items():
    if plot_name != "numerical_data":
        with open(f"{plot_name}.png", "wb") as f:
            f.write(base64.b64decode(base64_data))
# %%
