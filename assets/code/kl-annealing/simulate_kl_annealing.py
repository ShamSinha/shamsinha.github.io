"""Exact scalar-VAE simulation for visualizing KL weights and KL annealing."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("kl_annealing_outputs")
OUT.mkdir(exist_ok=True)

V = 4.0
RECON_WEIGHT = 0.05
BETA_TARGET = 0.38
STEPS = 3000
LR = 0.03
INITIAL = np.array([0.01, 0.01, 0.0], dtype=float)


def metrics(parameters: np.ndarray, beta: float) -> dict:
    a, b, log_variance = parameters
    variance = np.exp(log_variance)

    distortion = V * (1.0 - a * b) ** 2 + b**2 * variance
    kl = 0.5 * (V * a**2 + variance - log_variance - 1.0)
    mutual_information = 0.5 * np.log1p(V * a**2 / variance)
    objective = RECON_WEIGHT * distortion + beta * kl

    return {
        "distortion": distortion,
        "kl": kl,
        "mutual_information": mutual_information,
        "objective": objective,
        "variance": variance,
    }


def gradient(parameters: np.ndarray, beta: float) -> np.ndarray:
    a, b, log_variance = parameters
    variance = np.exp(log_variance)

    grad_a = (
        RECON_WEIGHT * (-2.0 * V * b * (1.0 - a * b))
        + beta * V * a
    )
    grad_b = RECON_WEIGHT * (
        -2.0 * V * a * (1.0 - a * b) + 2.0 * b * variance
    )
    grad_log_variance = (
        RECON_WEIGHT * b**2 * variance
        + 0.5 * beta * (variance - 1.0)
    )

    return np.array([grad_a, grad_b, grad_log_variance])


def simulate(name, beta_schedule):
    parameters = INITIAL.copy()
    rows = []

    for step in range(STEPS):
        beta = float(beta_schedule(step))
        values = metrics(parameters, beta)
        a, b, log_variance = parameters

        rows.append(
            {
                "experiment": name,
                "step": step,
                "beta": beta,
                "a": a,
                "b": b,
                "log_variance": log_variance,
                **values,
            }
        )

        parameters -= LR * gradient(parameters, beta)

    return pd.DataFrame(rows)


def save_line_plot(runs, column, ylabel, title, filename):
    fig = plt.figure(figsize=(8, 5.5))
    for name, frame in runs.items():
        plt.plot(frame["step"], frame[column], label=name)
    plt.xlabel("Optimization step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main():
    schedules = {
        "Constant beta = 0.38": lambda step: BETA_TARGET,
        "Warmup to beta = 0.38": lambda step: (
            BETA_TARGET * min(1.0, step / 1500.0)
        ),
        "Weak beta = 0.10": lambda step: 0.10,
        "Strong beta = 0.60": lambda step: 0.60,
    }

    runs = {
        name: simulate(name, schedule)
        for name, schedule in schedules.items()
    }
    history = pd.concat(runs.values(), ignore_index=True)
    history.to_csv(OUT / "training_history.csv", index=False)

    save_line_plot(
        runs,
        "beta",
        "Effective KL weight beta",
        "KL weighting schedules",
        "beta_schedules.png",
    )
    save_line_plot(
        runs,
        "mutual_information",
        "I(X; Z), nats",
        "Information carried by the latent variable",
        "information_during_training.png",
    )
    save_line_plot(
        runs,
        "distortion",
        "Expected squared reconstruction error",
        "Reconstruction during training",
        "reconstruction_during_training.png",
    )

    fig = plt.figure(figsize=(8, 6))
    for name, frame in runs.items():
        plt.plot(frame["a"], frame["b"], label=name)
    plt.scatter([INITIAL[0]], [INITIAL[1]], label="Initialization")
    plt.xlabel("Encoder gain a")
    plt.ylabel("Decoder reliance b")
    plt.title("Encoder-decoder coordination")
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUT / "encoder_decoder_trajectory.png", dpi=180)
    plt.close(fig)

    final = history.groupby("experiment").tail(1)[
        [
            "experiment",
            "beta",
            "distortion",
            "kl",
            "mutual_information",
            "a",
            "b",
            "variance",
        ]
    ]
    final.to_csv(OUT / "final_metrics.csv", index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
