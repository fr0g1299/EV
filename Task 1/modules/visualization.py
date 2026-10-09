"""Plot convergence curves produced by the binary genetic algorithm."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qbstyles import mpl_style

# Enable dark mode for matplotlib plots
mpl_style(dark=True, minor_ticks=False)


def plot_convergence(
    results: dict, dimensions: list[int], output_dir: str = "img/convergence"
):
    """Save one average best-so-far convergence plot per problem."""
    colors = ["#2563eb", "#ea580c", "#16a34a", "#dc2626", "#9333ea"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for problem, problem_results in results.items():
        _, ax = plt.subplots(figsize=(10, 7))
        for idx, D in enumerate(dimensions):
            convergence = np.asarray(problem_results[f"D{D}"]["avg_convergence"])
            fes_values = np.arange(1, len(convergence) + 1)
            ax.plot(
                fes_values,
                convergence,
                label=f"D = {D}",
                color=colors[idx % len(colors)],
                linewidth=2,
            )

        ax.set_xlabel("Function Evaluations (FES)", fontsize=12)
        ax.set_ylabel("Average best fitness", fontsize=12)
        ax.set_title(f"Convergence Curve: {problem}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.legend(fontsize=11)
        plt.tight_layout()
        filename = output_path / f"convergence_{problem}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
        plt.close()
