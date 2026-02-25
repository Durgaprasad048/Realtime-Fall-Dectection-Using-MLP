import csv
import math
import os
from pathlib import Path

import numpy as np

OUT_DIR = Path("./Example Datasets/Example Datasets/Realistic Synthetic")
SAMPLE_RATE_HZ = 50.0
DT = 1.0 / SAMPLE_RATE_HZ
N_SAMPLES = 180

ACTIVITIES = ["STD", "WAL", "JOG", "JUM", "FALL", "LYI", "RA"]
SUBJECTS = ["E", "F", "G"]
SESSIONS = [1, 2]


def subject_params(rng: np.random.Generator):
    return {
        "gravity": 9.81 + rng.normal(0, 0.12),
        "acc_noise": 0.10 + rng.uniform(0.00, 0.05),
        "gyro_noise": 0.035 + rng.uniform(0.00, 0.02),
        "stride_scale": 1.0 + rng.uniform(-0.15, 0.15),
        "arm_swing_scale": 1.0 + rng.uniform(-0.20, 0.20),
    }


def base_orientation(gravity: float, mode: str):
    if mode == "lying":
        return np.array([0.2, -0.3, gravity], dtype=float)
    tilt_x = -gravity * 0.75
    tilt_y = -gravity * 0.65
    tilt_z = gravity * 0.05
    return np.array([tilt_x, tilt_y, tilt_z], dtype=float)


def generate_activity(activity: str, seed: int):
    rng = np.random.default_rng(seed)
    p = subject_params(rng)
    t = np.arange(N_SAMPLES) * DT

    standing_g = base_orientation(p["gravity"], mode="standing")
    lying_g = base_orientation(p["gravity"], mode="lying")

    acc = np.tile(standing_g, (N_SAMPLES, 1))
    gyro = np.zeros((N_SAMPLES, 3), dtype=float)

    if activity == "STD":
        sway = 0.15 * np.sin(2 * math.pi * 0.18 * t + rng.uniform(0, 2 * math.pi))
        acc[:, 0] += sway
        acc[:, 1] += 0.10 * np.sin(2 * math.pi * 0.11 * t)
        gyro[:, 2] += 0.12 * np.sin(2 * math.pi * 0.22 * t)
    elif activity == "WAL":
        f = 1.75 * p["stride_scale"]
        step = np.sin(2 * math.pi * f * t)
        acc[:, 2] += 1.25 * np.maximum(step, -0.4)
        acc[:, 0] += 0.65 * np.sin(2 * math.pi * f * t + 1.2)
        acc[:, 1] += 0.35 * np.sin(2 * math.pi * f * t + 0.4)
        gyro[:, 1] += 0.9 * np.sin(2 * math.pi * f * t)
        gyro[:, 2] += 0.55 * np.sin(2 * math.pi * 2 * f * t)
    elif activity == "JOG":
        f = 2.5 * p["stride_scale"]
        step = np.sin(2 * math.pi * f * t)
        acc[:, 2] += 2.4 * np.maximum(step, -0.3)
        acc[:, 0] += 1.2 * np.sin(2 * math.pi * f * t + 1.1)
        acc[:, 1] += 0.8 * np.sin(2 * math.pi * f * t + 0.2)
        gyro[:, 0] += 1.1 * np.sin(2 * math.pi * f * t + 0.7)
        gyro[:, 1] += 1.5 * np.sin(2 * math.pi * f * t)
        gyro[:, 2] += 0.9 * np.sin(2 * math.pi * 2 * f * t + 0.3)
    elif activity == "JUM":
        acc[:, :] = np.tile(standing_g, (N_SAMPLES, 1))
        pulse_positions = [35, 85, 135]
        for center in pulse_positions:
            width = 8
            idx = np.arange(N_SAMPLES)
            takeoff = np.exp(-((idx - (center - width)) ** 2) / (2 * (width / 2.2) ** 2))
            landing = np.exp(-((idx - (center + width)) ** 2) / (2 * (width / 1.8) ** 2))
            acc[:, 2] += 3.4 * takeoff + 5.5 * landing
            gyro[:, 0] += 1.9 * takeoff
            gyro[:, 1] += 1.5 * landing
    elif activity == "RA":
        f = 2.2 * p["arm_swing_scale"]
        swing = np.sin(2 * math.pi * f * t)
        acc[:, 0] += 0.95 * swing
        acc[:, 1] += 0.45 * np.sin(2 * math.pi * f * t + 0.8)
        acc[:, 2] += 0.65 * np.sin(2 * math.pi * 2 * f * t + 0.4)
        gyro[:, 0] += 2.8 * swing
        gyro[:, 1] += 2.0 * np.sin(2 * math.pi * f * t + 1.0)
        gyro[:, 2] += 1.3 * np.sin(2 * math.pi * 2 * f * t)
    elif activity == "LYI":
        acc[:, :] = np.tile(lying_g, (N_SAMPLES, 1))
        resp = 0.08 * np.sin(2 * math.pi * 0.24 * t)
        acc[:, 2] += resp
        gyro[:, 2] += 0.06 * np.sin(2 * math.pi * 0.2 * t)
    elif activity == "FALL":
        # Start with normal standing micro-motion.
        acc[:, :] = np.tile(standing_g, (N_SAMPLES, 1))
        acc[:, 0] += 0.12 * np.sin(2 * math.pi * 0.3 * t)
        acc[:, 1] += 0.10 * np.sin(2 * math.pi * 0.25 * t)

        idx = np.arange(N_SAMPLES)
        pre = np.exp(-((idx - 70) ** 2) / (2 * 6.0 ** 2))
        impact = np.exp(-((idx - 86) ** 2) / (2 * 2.2 ** 2))
        settle = np.exp(-((idx - 100) ** 2) / (2 * 6.5 ** 2))

        acc[:, 0] += 4.0 * pre + 11.0 * impact
        acc[:, 1] -= 3.0 * pre + 8.0 * impact
        acc[:, 2] += 2.0 * pre + 14.0 * impact

        gyro[:, 0] += 5.5 * pre + 12.0 * impact
        gyro[:, 1] -= 4.0 * pre + 9.0 * impact
        gyro[:, 2] += 3.0 * pre + 7.0 * impact

        # Transition orientation to lying after impact.
        alpha = np.clip((idx - 92) / 22.0, 0.0, 1.0)
        target = np.tile(lying_g, (N_SAMPLES, 1))
        acc = (1.0 - alpha[:, None]) * acc + alpha[:, None] * target
        gyro *= np.clip(1.3 - settle[:, None], 0.12, 1.0)

    acc += rng.normal(0, p["acc_noise"], size=acc.shape)
    gyro += rng.normal(0, p["gyro_noise"], size=gyro.shape)

    return acc, gyro


def write_csv(path: Path, acc: np.ndarray, gyro: np.ndarray, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label"])
        for i in range(len(acc)):
            writer.writerow(
                [
                    round(float(acc[i, 0]), 3),
                    round(float(acc[i, 1]), 3),
                    round(float(acc[i, 2]), 3),
                    round(float(gyro[i, 0]), 3),
                    round(float(gyro[i, 1]), 3),
                    round(float(gyro[i, 2]), 3),
                    label,
                ]
            )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for subject in SUBJECTS:
        for session in SESSIONS:
            for activity in ACTIVITIES:
                seed = abs(hash((subject, session, activity))) % (2**32)
                acc, gyro = generate_activity(activity, seed)
                fname = f"{activity}_Subject{subject}_{session:02d}.csv"
                out_file = OUT_DIR / fname
                write_csv(out_file, acc, gyro, activity)
                created.append(out_file)

    print(f"Generated {len(created)} files in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
