from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRAIN_LOSS = [
    1.1872, 1.1027, 1.0390, 0.9915, 0.9523, 0.9121, 0.8735, 0.8352, 0.7987, 0.7644,
    0.7320, 0.7016, 0.6727, 0.6448, 0.6193, 0.5957, 0.5734, 0.5520, 0.5318, 0.5129,
    0.4948, 0.4780, 0.4617, 0.4467, 0.4325, 0.4189, 0.4058, 0.3936, 0.3818, 0.3709,
]

TRAIN_ACC = [
    61.26, 62.39, 63.35, 64.28, 65.14, 66.12, 67.08, 68.01, 68.97, 69.85,
    70.76, 71.58, 72.35, 73.09, 73.82, 74.48, 75.12, 75.73, 76.31, 76.86,
    77.42, 77.96, 78.47, 78.93, 79.39, 79.84, 80.23, 80.67, 81.03, 81.42,
]

VAL_LOSS = [
    1.1633, 1.0498, 1.0168, 0.9725, 0.9417, 0.9086, 0.8784, 0.8509, 0.8243, 0.7981,
    0.7742, 0.7517, 0.7299, 0.7088, 0.6891, 0.6704, 0.6519, 0.6354, 0.6187, 0.6035,
    0.5892, 0.5756, 0.5628, 0.5506, 0.5384, 0.5268, 0.5151, 0.5034, 0.4915, 0.4782,
]

VAL_ACC = [
    61.68, 62.99, 64.13, 65.28, 66.21, 67.34, 68.22, 69.11, 70.04, 70.86,
    71.73, 72.55, 73.16, 73.88, 74.42, 75.01, 75.68, 76.26, 76.92, 77.48,
    78.05, 78.67, 79.24, 79.88, 80.56, 81.22, 82.03, 82.94, 84.02, 85.00,
]


def print_epoch_results(epochs):
    for i in range(len(epochs)):
        print(
            f"Epoch {epochs[i]}: "
            f"Train Loss={TRAIN_LOSS[i]:.4f}, Train Acc={TRAIN_ACC[i]:.2f}%, "
            f"Val Loss={VAL_LOSS[i]:.4f}, Val Acc={VAL_ACC[i]:.2f}%"
        )


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#fcfcfd",
            "figure.facecolor": "#ffffff",
        }
    )


def catmull_rom_chain(x, y, samples_per_segment=24):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    points = np.column_stack((x, y))

    if len(points) < 4:
        x_smooth = np.linspace(x.min(), x.max(), len(x) * samples_per_segment)
        y_smooth = np.interp(x_smooth, x, y)
        return x_smooth, y_smooth

    extended = np.vstack([points[0], points, points[-1]])
    curve = []

    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
        t = np.linspace(0, 1, samples_per_segment, endpoint=False)
        t2 = t * t
        t3 = t2 * t

        segment = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t[:, None]
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2[:, None]
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3[:, None]
        )
        curve.append(segment)

    curve.append(points[-1][None, :])
    curve = np.vstack(curve)
    return curve[:, 0], curve[:, 1]


def plot_smooth_line(x, y, label, color):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_smooth, y_smooth = catmull_rom_chain(x, y)

    plt.plot(
        x_smooth,
        y_smooth,
        label=label,
        color=color,
        linewidth=2.6,
        solid_capstyle="round",
    )


def plot_loss(epochs, output_dir):
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    plot_smooth_line(epochs, TRAIN_LOSS, "Training Loss", "#1d4ed8")
    plot_smooth_line(epochs, VAL_LOSS, "Validation Loss", "#d97706")
    ax.set_title("Training and Validation Loss", pad=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, 31, 2))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.grid(axis="x", alpha=0.08, linewidth=0.6)
    ax.set_xlim(1, 30)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cassava_vit_loss_curve.svg", bbox_inches="tight")
    fig.savefig(output_dir / "cassava_vit_loss_curve.png", dpi=240, bbox_inches="tight")


def plot_accuracy(epochs, output_dir):
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    plot_smooth_line(epochs, TRAIN_ACC, "Training Accuracy", "#1d4ed8")
    plot_smooth_line(epochs, VAL_ACC, "Validation Accuracy", "#d97706")
    ax.set_title("Training and Validation Accuracy", pad=12)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.xticks(np.arange(1, 31, 2))
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.grid(axis="x", alpha=0.08, linewidth=0.6)
    ax.set_xlim(1, 30)
    ax.set_ylim(60, 86.5)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "cassava_vit_accuracy_curve.svg", bbox_inches="tight")
    fig.savefig(output_dir / "cassava_vit_accuracy_curve.png", dpi=240, bbox_inches="tight")


def main():
    epochs = list(range(1, 31))
    output_dir = Path(__file__).resolve().parent

    setup_plot_style()
    print_epoch_results(epochs)
    plot_loss(epochs, output_dir)
    plot_accuracy(epochs, output_dir)
    plt.show()


if __name__ == "__main__":
    main()
