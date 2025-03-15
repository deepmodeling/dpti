# %%
import base64
import glob
import json
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata


class GibbsSurfaceBuilder:
    """
    Class for building and visualizing Gibbs free energy surfaces
    from thermodynamic integration simulation results.
    """

    def __init__(self):
        self.data_points = []  # List to store (temp, press, fe) data points
        self.phase_data = {}  # Dictionary to store data by phase

    def load_result_file(
        self, file_path: str, phase_name: Optional[str] = None
    ) -> Dict:
        """
        Load a single result file and extract temperature, pressure, and free energy data.

        Args:
            file_path: Path to the result JSON file
            phase_name: Optional name of the phase (e.g., "solid", "liquid")

        Returns
        -------
            Dictionary containing the loaded data
        """
        try:
            with open(file_path) as f:
                result_info = json.load(f)

            # Extract the necessary data
            temps = np.array(result_info["data"]["all_temps"])
            press = np.array(result_info["data"]["all_press"])
            fe = np.array(result_info["data"]["all_fe"])

            # Optional: extract error data if available
            fe_stat_err = np.array(
                result_info["data"].get("all_fe_stat_err", [0] * len(temps))
            )
            fe_inte_err = np.array(
                result_info["data"].get("all_fe_inte_err", [0] * len(temps))
            )
            total_err = np.sqrt(fe_stat_err**2 + fe_inte_err**2)

            data = {
                "temps": temps,
                "press": press,
                "fe": fe,
                "total_err": total_err,
                "meta": {"file_path": file_path, "phase": phase_name},
            }

            # Add to data points collection
            for t, p, f in zip(temps, press, fe):
                self.data_points.append((t, p, f))

            # Store by phase if provided
            if phase_name:
                if phase_name not in self.phase_data:
                    self.phase_data[phase_name] = []
                self.phase_data[phase_name].append(data)

            return data

        except Exception as e:
            print(f"Error loading file {file_path}: {e!s}")
            return None

    def load_multiple_files(
        self, file_pattern: str, phase_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Load multiple result files using a glob pattern.

        Args:
            file_pattern: Glob pattern to match files
            phase_name: Optional name of the phase

        Returns
        -------
            List of loaded data dictionaries
        """
        files = glob.glob(file_pattern)
        results = []

        for file_path in files:
            result = self.load_result_file(file_path, phase_name)
            if result:
                results.append(result)

        print(f"Loaded {len(results)} files matching pattern: {file_pattern}")
        return results

    def build_surface(
        self,
        temp_range: Optional[Tuple[float, float]] = None,
        press_range: Optional[Tuple[float, float]] = None,
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a Gibbs free energy surface from the collected data points.

        Args:
            temp_range: Optional tuple (min_temp, max_temp) to limit range
            press_range: Optional tuple (min_press, max_press) to limit range
            resolution: Number of points in the interpolated grid

        Returns
        -------
            Tuple of (temp_grid, press_grid, fe_grid) arrays for plotting
        """
        if not self.data_points:
            raise ValueError("No data points available. Load data first.")

        # Convert to numpy arrays
        points = np.array([(t, p) for t, p, _ in self.data_points])
        values = np.array([f for _, _, f in self.data_points])

        # Determine ranges if not provided
        if temp_range is None:
            temp_min = np.min(points[:, 0])
            temp_max = np.max(points[:, 0])
            temp_range = (temp_min, temp_max)
        else:
            temp_min, temp_max = temp_range

        if press_range is None:
            press_min = np.min(points[:, 1])
            press_max = np.max(points[:, 1])
            press_range = (press_min, press_max)
        else:
            press_min, press_max = press_range

        # Create grid for interpolation
        temp_grid = np.linspace(temp_min, temp_max, resolution)
        press_grid = np.linspace(press_min, press_max, resolution)
        T, P = np.meshgrid(temp_grid, press_grid)

        # Interpolate free energy values on grid
        fe_grid = griddata(points, values, (T, P), method="cubic", fill_value=np.nan)

        return T, P, fe_grid

    def plot_surface(
        self,
        T: np.ndarray,
        P: np.ndarray,
        fe_grid: np.ndarray,
        title: str = "Gibbs Free Energy Surface",
    ) -> str:
        """
        Plot the Gibbs free energy surface as a 3D surface.

        Args:
            T: Temperature grid
            P: Pressure grid
            fe_grid: Free energy grid
            title: Plot title

        Returns
        -------
            Base64 encoded string of the plot image
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Create the surface plot
        surf = ax.plot_surface(
            T, P, fe_grid, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8
        )

        # Add a color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Free Energy (eV/atom)")

        # Add the original data points
        data_points = np.array(self.data_points)
        ax.scatter(
            data_points[:, 0],
            data_points[:, 1],
            data_points[:, 2],
            color="r",
            s=30,
            label="Data Points",
        )

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Pressure (bar)")
        ax.set_zlabel("Gibbs Free Energy (eV/atom)")
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()

        # Save the figure to a base64 string
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def plot_contour(
        self,
        T: np.ndarray,
        P: np.ndarray,
        fe_grid: np.ndarray,
        title: str = "Gibbs Free Energy Contour",
    ) -> str:
        """
        Plot the Gibbs free energy surface as a 2D contour map.

        Args:
            T: Temperature grid
            P: Pressure grid
            fe_grid: Free energy grid
            title: Plot title

        Returns
        -------
            Base64 encoded string of the plot image
        """
        fig = plt.figure(figsize=(12, 8))

        # Create the contour plot
        contour = plt.contourf(T, P, fe_grid, 50, cmap="viridis")
        plt.colorbar(contour, label="Free Energy (eV/atom)")

        # Add contour lines
        contour_lines = plt.contour(
            T, P, fe_grid, 10, colors="white", alpha=0.5, linewidths=0.8
        )
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.4f")

        # Add the original data points
        data_points = np.array(self.data_points)
        plt.scatter(
            data_points[:, 0],
            data_points[:, 1],
            color="red",
            s=30,
            marker="o",
            label="Data Points",
        )

        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (bar)")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()

        # Save the figure to a base64 string
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def find_phase_boundary(
        self,
        phase1: str,
        phase2: str,
        temp_range: Optional[Tuple[float, float]] = None,
        press_range: Optional[Tuple[float, float]] = None,
        resolution: int = 100,
    ) -> dict:
        """
        Find the phase boundary between two phases.

        Args:
            phase1: Name of first phase
            phase2: Name of second phase
            temp_range: Optional temperature range
            press_range: Optional pressure range
            resolution: Resolution of the grid

        Returns
        -------
            Dictionary with phase boundary data and plot
        """
        if phase1 not in self.phase_data or phase2 not in self.phase_data:
            raise ValueError(
                f"Missing phase data. Available phases: {list(self.phase_data.keys())}"
            )

        # Extract and merge data points for each phase
        phase1_points = []
        phase1_values = []

        for data in self.phase_data[phase1]:
            temps = data["temps"]
            press = data["press"]
            fe = data["fe"]

            for t, p, f in zip(temps, press, fe):
                phase1_points.append((t, p))
                phase1_values.append(f)

        phase2_points = []
        phase2_values = []

        for data in self.phase_data[phase2]:
            temps = data["temps"]
            press = data["press"]
            fe = data["fe"]

            for t, p, f in zip(temps, press, fe):
                phase2_points.append((t, p))
                phase2_values.append(f)

        # Convert to numpy arrays
        phase1_points = np.array(phase1_points)
        phase1_values = np.array(phase1_values)
        phase2_points = np.array(phase2_points)
        phase2_values = np.array(phase2_values)

        # Determine ranges if not provided
        if temp_range is None:
            temp_min = min(np.min(phase1_points[:, 0]), np.min(phase2_points[:, 0]))
            temp_max = max(np.max(phase1_points[:, 0]), np.max(phase2_points[:, 0]))
            temp_range = (temp_min, temp_max)
        else:
            temp_min, temp_max = temp_range

        if press_range is None:
            press_min = min(np.min(phase1_points[:, 1]), np.min(phase2_points[:, 1]))
            press_max = max(np.max(phase1_points[:, 1]), np.max(phase2_points[:, 1]))
            press_range = (press_min, press_max)
        else:
            press_min, press_max = press_range

        # Create grid for interpolation
        temp_grid = np.linspace(temp_min, temp_max, resolution)
        press_grid = np.linspace(press_min, press_max, resolution)
        T, P = np.meshgrid(temp_grid, press_grid)

        # Interpolate free energy values on grid for both phases
        fe_grid1 = griddata(
            phase1_points, phase1_values, (T, P), method="cubic", fill_value=np.nan
        )
        fe_grid2 = griddata(
            phase2_points, phase2_values, (T, P), method="cubic", fill_value=np.nan
        )

        # Calculate free energy difference
        fe_diff = fe_grid2 - fe_grid1

        # Find phase boundary (where fe_diff = 0)
        phase_boundary = np.zeros_like(fe_diff)
        phase_boundary[np.abs(fe_diff) < 0.0001] = (
            1  # tolerance for numerical stability
        )

        # Extract boundary points
        boundary_points = []
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                if phase_boundary[i, j] == 1:
                    boundary_points.append((T[i, j], P[i, j]))

        # Plot the phase diagram
        fig = plt.figure(figsize=(12, 10))

        # Plot free energy difference as a contour
        levels = np.linspace(np.nanmin(fe_diff), np.nanmax(fe_diff), 50)
        contour = plt.contourf(T, P, fe_diff, levels=levels, cmap="RdBu_r", alpha=0.7)
        plt.colorbar(contour, label="Energy Difference (eV/atom)")

        # Add the zero contour line to indicate phase boundary
        zero_contour = plt.contour(T, P, fe_diff, levels=[0], colors="k", linewidths=2)

        # Add data points
        plt.scatter(
            phase1_points[:, 0],
            phase1_points[:, 1],
            color="blue",
            s=30,
            marker="o",
            label=phase1,
        )
        plt.scatter(
            phase2_points[:, 0],
            phase2_points[:, 1],
            color="red",
            s=30,
            marker="o",
            label=phase2,
        )

        plt.xlabel("Temperature (K)")
        plt.ylabel("Pressure (bar)")
        plt.title(f"Phase Boundary between {phase1} and {phase2}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()

        # Save the figure to a base64 string
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        boundary_plot = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "boundary_points": boundary_points,
            "plot": boundary_plot,
            "fe_diff": fe_diff.tolist(),
            "T": T.tolist(),
            "P": P.tolist(),
        }

    def save_data(self, output_file: str) -> None:
        """
        Save the collected data points to a JSON file.

        Args:
            output_file: Output file path
        """
        output_data = {"data_points": self.data_points, "phase_data": {}}

        # Convert phase data to serializable format
        for phase, data_list in self.phase_data.items():
            output_data["phase_data"][phase] = []
            for data in data_list:
                serializable_data = {
                    "temps": data["temps"].tolist(),
                    "press": data["press"].tolist(),
                    "fe": data["fe"].tolist(),
                    "total_err": data["total_err"].tolist(),
                    "meta": data["meta"],
                }
                output_data["phase_data"][phase].append(serializable_data)

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Data saved to {output_file}")

    def plot_interpolated_surface(
        self,
        temp_range: Optional[Tuple[float, float]] = None,
        press_range: Optional[Tuple[float, float]] = None,
        resolution: int = 100,
        interpolation_method: str = "cubic",
    ) -> str:
        """
        展示插值后的自由能曲面（包含原始数据点）.

        Args:
            temp_range: 温度范围 (min, max)
            press_range: 压力范围 (min, max)
            resolution: 网格分辨率
            interpolation_method: 插值方法 ('linear', 'cubic', 'nearest')

        Returns
        -------
            Base64编码的曲面图
        """
        # 构建插值曲面
        T, P, fe_grid = self.build_surface(temp_range, press_range, resolution)

        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # 绘制插值曲面
        surf = ax.plot_surface(
            T,
            P,
            fe_grid,
            cmap=cm.viridis,
            rstride=2,
            cstride=2,  # 控制网格密度
            linewidth=0.5,  # 网格线宽
            edgecolor="grey",  # 网格线颜色
            alpha=0.8,
        )  # 曲面透明度

        # 添加原始数据点
        data_points = np.array(self.data_points)
        ax.scatter(
            data_points[:, 0],
            data_points[:, 1],
            data_points[:, 2],
            color="red",
            s=50,
            edgecolor="k",
            label="Raw Data Points",
            zorder=10,
        )

        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label("Free Energy (eV/atom)", rotation=270, labelpad=15)

        # 设置轴标签和标题
        ax.set_xlabel("Temperature (K)", labelpad=12)
        ax.set_ylabel("Pressure (bar)", labelpad=12)
        ax.set_zlabel("Gibbs Free Energy (eV/atom)", labelpad=12)
        ax.set_title(
            f"Interpolated Gibbs Surface ({interpolation_method.capitalize()} Interpolation)",
            pad=20,
        )

        # 优化视角
        ax.view_init(elev=35, azim=45)  # 调整视角角度

        # 添加图例
        ax.legend(loc="upper right", fontsize=10)

        plt.show()
        # 保存为Base64
        buf = BytesIO()

        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def save_data_points(self, output_path: Optional[str] = None) -> str:
        """
        将加载的数据点保存为格式化文本文件.

        Args:
            output_path: 可选输出路径，默认为当前目录

        Returns
        -------
            保存的文件路径
        """
        if not self.data_points:
            raise ValueError("没有可用的数据点，请先加载数据")

        # 转换为numpy数组并排序
        data_array = np.array(self.data_points)
        data_array = data_array[
            np.lexsort((data_array[:, 1], data_array[:, 0]))
        ]  # 按温度、压力排序

        # 生成默认文件名
        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"gibbs_data_points_{timestamp}.txt"

        # 自定义格式模板
        fmt = [
            "%12.3f",  # 温度：固定12字符宽度，3位小数
            "%12.3f",  # 压力：固定12字符宽度，3位小数
            "%18.8f",  # 自由能：固定18字符宽度，8位小数
        ]

        # 保存为对齐的文本文件
        np.savetxt(
            output_path,
            data_array,
            fmt=fmt,
            delimiter="    ",  # 4空格分隔
            header="Temperature (K)    Pressure (bar)    Gibbs Free Energy (eV/atom)",
            comments="# ",
            encoding="utf-8",
        )

        print(f"成功保存 {len(data_array)} 个数据点到 {output_path}")
        return output_path


def main():
    # Example usage
    builder = GibbsSurfaceBuilder()

    # Load solid phase data
    solid_pattern = "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_*K_*bar_run*/TI_sim/new_job/result.json"
    builder.load_multiple_files(solid_pattern, phase_name="solid")

    # Load liquid phase data
    liquid_pattern = "../examples/Sn_beta_quicktest/TI_liquid128_path_t_*K_*bar_run*/TI_sim/new_job/result.json"
    builder.load_multiple_files(liquid_pattern, phase_name="liquid")

    # Build surface
    T, P, fe_grid = builder.build_surface()

    # Generate plots
    surface_plot_base64 = builder.plot_surface(
        T, P, fe_grid, title="Gibbs Free Energy Surface"
    )
    contour_plot_base64 = builder.plot_contour(
        T, P, fe_grid, title="Gibbs Free Energy Contour"
    )

    # Find phase boundary
    try:
        boundary_data = builder.find_phase_boundary("solid", "liquid")
        print(f"Found {len(boundary_data['boundary_points'])} boundary points")
    except ValueError as e:
        print(f"Could not find phase boundary: {e!s}")

    # Save data
    builder.save_data("gibbs_surface_data.json")

    # Save plots if needed
    with open("surface_plot.png", "wb") as f:
        f.write(base64.b64decode(surface_plot_base64))

    with open("contour_plot.png", "wb") as f:
        f.write(base64.b64decode(contour_plot_base64))


# %%
if __name__ == "__main__":
    main()

# %%

builder = GibbsSurfaceBuilder()

# 加载数据文件
builder.load_multiple_files("你的文件路径模式*.json", phase_name="相名称")

# 构建自由能面
T, P, fe_grid = builder.build_surface()

# 生成可视化
surface_plot = builder.plot_surface(T, P, fe_grid)
contour_plot = builder.plot_contour(T, P, fe_grid)

# 查找相界
boundary_data = builder.find_phase_boundary("相1", "相2")

# 保存数据
builder.save_data("输出文件.json")


# %%


# 指定要读取的文件路径
file_path = "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run3/TI_sim/new_job/result.json"


builder = GibbsSurfaceBuilder()

# 加载文件并获取结果
result_data = builder.load_result_file(file_path, phase_name="solid")

# 打印数据点
if result_data:
    print("成功加载文件。发现以下数据点:")
    print("\n温度(K)\t压力(bar)\t自由能(eV/atom)")
    print("-" * 50)

    # 遍历所有数据点并打印
    for t, p, f in zip(result_data["temps"], result_data["press"], result_data["fe"]):
        print(f"{t:.2f}\t{p:.2f}\t\t{f:.6f}")

    # 绘制一个简单的图表展示数据
    plt.figure(figsize=(10, 6))
    plt.scatter(
        result_data["temps"],
        result_data["fe"],
        c=result_data["press"],
        cmap="viridis",
        s=50,
        alpha=0.8,
    )

    plt.colorbar(label="压力 (bar)")
    plt.xlabel("温度 (K)")
    plt.ylabel("自由能 (eV/atom)")
    plt.title("从JSON文件读取的自由能数据")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # 如果您只想直接访问原始数据
    print("\n您可以直接访问以下数据:")
    print(f"温度数组: {result_data['temps']}")
    print(f"压力数组: {result_data['press']}")
    print(f"自由能数组: {result_data['fe']}")
else:
    print(f"加载文件时出错: {file_path}")


# %%

# %%
# 导入必要的库

file_path = "../examples/Sn_beta_quicktest/TI_beta144xy_path_t_200K_50000bar_run3/TI_sim/new_job/result.json"

# 读取JSON文件
with open(file_path) as f:
    result_info = json.load(f)

# 获取数据
temps = np.array(result_info["data"]["all_temps"])
press = np.array(result_info["data"]["all_press"])
fe = np.array(result_info["data"]["all_fe"])

# 创建3D图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# 绘制3D散点图 - 使用自由能值来设置颜色
scatter = ax.scatter(
    temps, press, fe, c=fe, cmap="viridis", s=100, marker="o", edgecolors="k", alpha=0.8
)

# 添加颜色条
colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label="自由能 (eV/atom)")

# 设置轴标签
ax.set_xlabel("温度 (K)", fontsize=14)
ax.set_ylabel("压力 (bar)", fontsize=14)
ax.set_zlabel("自由能 (eV/atom)", fontsize=14)
ax.set_title("自由能3D散点图", fontsize=16)

# 优化视角
ax.view_init(elev=30, azim=45)  # 设置视角角度

# 显示图形
plt.tight_layout()
plt.show()

# %%
# 创建GibbsSurfaceBuilder实例
builder = GibbsSurfaceBuilder()

# 加载多个文件
# file_pattern = "../examples/Sn_beta_quicktest/TI_beta144xy_path_?_*K_*bar_run*/TI_sim/new_job/result.json"
builder = GibbsSurfaceBuilder()
file_pattern = (
    "../examples/Sn_beta_quicktest/TI_beta144xy_path*/TI_sim/new_job/result.json"
)

# %%
builder = GibbsSurfaceBuilder()
file_pattern = (
    "../examples/Sn_beta_quicktest/TI_liquid128_path*/TI_sim/new_job/result.json"
)

# %%
builder.load_multiple_files(file_pattern, phase_name="solid")


saved_path = builder.save_data_points()
print(f"数据点已保存到: {saved_path}")
# %%


# %%
# 加载另一组数据，例如液相数据
# liquid_pattern = "../examples/Sn_beta_quicktest/TI_liquid128_path_t_*K_*bar_run*/TI_sim/new_job/result.json"
# builder.load_multiple_files(liquid_pattern, phase_name="liquid")

# 将数据点转换为数组
data_points = np.array(builder.data_points)
temps = data_points[:, 0]
press = data_points[:, 1]
fe = data_points[:, 2]

# 创建具有相信息的颜色映射
phase_colors = []
for phase_name, phase_data in builder.phase_data.items():
    for data in phase_data:
        for _ in range(len(data["temps"])):
            phase_colors.append(phase_name)

# 创建3D图形
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

# 为不同相设置不同颜色
phase_color_map = {"solid": "blue", "liquid": "red"}

# 绘制所有相的3D散点图
for phase, color in phase_color_map.items():
    if phase in builder.phase_data:
        # 找到属于当前相的点
        phase_indices = [i for i, p in enumerate(phase_colors) if p == phase]
        if phase_indices:
            phase_temps = temps[phase_indices]
            phase_press = press[phase_indices]
            phase_fe = fe[phase_indices]

            # 绘制该相的散点
            ax.scatter(
                phase_temps,
                phase_press,
                phase_fe,
                c=color,
                s=80,
                label=phase,
                alpha=0.7,
                edgecolors="k",
            )

# 设置轴标签
ax.set_xlabel("温度 (K)", fontsize=14)
ax.set_ylabel("压力 (bar)", fontsize=14)
ax.set_zlabel("自由能 (eV/atom)", fontsize=14)
ax.set_title("不同相的自由能3D散点图", fontsize=16)

# 添加图例
ax.legend(fontsize=12)

# 优化视角
ax.view_init(elev=30, azim=45)

# 显示图形
plt.tight_layout()
plt.show()

# %%
surface_plot = builder.plot_interpolated_surface(
    temp_range=(200, 1000),  # 指定温度范围
    press_range=(0, 100000),  # 指定压力范围
    resolution=150,  # 提高网格分辨率
    interpolation_method="cubic",  # 选择插值方法
)
# %%
