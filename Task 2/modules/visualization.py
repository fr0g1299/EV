"""Plot convergence curves produced by the real-valued genetic algorithm."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from qbstyles import mpl_style

# Enable dark mode for matplotlib plots
mpl_style(dark=True, minor_ticks=False)

VARIANT_DISPLAY_NAMES = {
    "ieee754": "IEEE 754 Binary",
    "fixed_point": "Fixed-Point Binary",
    "bcd": "BCD Binary",
    "real_gaussian": "Real (Gaussian Mut.)",
    "real_uniform": "Real (Uniform Mut.)",
}


def plot_convergence(results: dict, output_dir: str = "img/convergence"):
    """Save one average best-so-far convergence plot per benchmark function.

    Parameters
    ----------
    results : dict
        Nested dictionary keyed by function name, then variant name.
        Each variant entry must contain an "avg_convergence" list.
    output_dir : str, optional
        Directory where the PNG files are saved (created if needed).
    """
    colors = ["#2563eb", "#ea580c", "#16a34a", "#dc2626", "#9333ea"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for function_name, variants in results.items():
        _, ax = plt.subplots(figsize=(10, 7))

        for idx, (variant_key, variant_data) in enumerate(variants.items()):
            convergence = np.asarray(variant_data["avg_convergence"])
            fes_values = np.arange(1, len(convergence) + 1)
            label = VARIANT_DISPLAY_NAMES.get(variant_key, variant_key)
            ax.plot(
                fes_values,
                convergence,
                label=label,
                color=colors[idx % len(colors)],
                linewidth=2,
            )

        ax.set_xlabel("Function Evaluations (FES)", fontsize=12)
        ax.set_ylabel("Average Best Fitness", fontsize=12)
        ax.set_title(
            f"Convergence: {function_name.capitalize()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.legend(fontsize=11)
        plt.tight_layout()
        filename = output_path / f"convergence_{function_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
        plt.close()
