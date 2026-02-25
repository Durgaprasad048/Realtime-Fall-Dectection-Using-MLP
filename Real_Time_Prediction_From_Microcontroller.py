import socket
import os
import sys
from collections import deque

EXTRA_SITE_PACKAGES = r"C:\tfpkgs"
if os.path.isdir(EXTRA_SITE_PACKAGES) and EXTRA_SITE_PACKAGES not in sys.path:
    sys.path.insert(0, EXTRA_SITE_PACKAGES)

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(
        "TensorFlow is unavailable or broken in this environment. "
        "Reinstall TensorFlow and retry.\n"
        f"Original error: {exc}"
    )

STEP_SIZE = 20
SENSOR_NUM = 6

MODEL_DIR = "./model_x"
MODEL_FILE = "./model_x.keras"
HOST = "0.0.0.0"
PORT = 80

CLASS_NAMES = {0: "STD", 1: "WAL", 2: "JOG", 3: "JUM", 4: "FALL", 5: "LYI", 6: "RA"}


class LivePlotter:
    def __init__(self, max_points=200):
        self.max_points = max_points
        self.sample_idx = 0
        self.x_data = deque(maxlen=max_points)
        self.acc_x = deque(maxlen=max_points)
        self.acc_y = deque(maxlen=max_points)
        self.acc_z = deque(maxlen=max_points)
        self.gyro_x = deque(maxlen=max_points)
        self.gyro_y = deque(maxlen=max_points)
        self.gyro_z = deque(maxlen=max_points)
        self.acc_mag_data = deque(maxlen=max_points)
        self.gyro_mag_data = deque(maxlen=max_points)
        self.fall_prob_data = deque(maxlen=max_points)
        self.pred_class_data = deque(maxlen=max_points)
        self.latest_pred_text = "Prediction: -"

        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        self.ax_sensor = axes[0, 0]
        self.ax_axes = axes[0, 1]
        self.ax_probs = axes[1, 0]
        self.ax_fall = axes[1, 1]
        self.fig.suptitle("Realtime Fall Detection")

        self.acc_line, = self.ax_sensor.plot([], [], label="|acc|", color="tab:blue")
        self.gyro_line, = self.ax_sensor.plot([], [], label="|gyro|", color="tab:orange")
        self.ax_sensor.set_title("Sensor Magnitude")
        self.ax_sensor.set_xlabel("Sample")
        self.ax_sensor.set_ylabel("Magnitude")
        self.ax_sensor.legend(loc="upper right")
        self.ax_sensor.grid(alpha=0.3)

        self.acc_x_line, = self.ax_axes.plot([], [], label="acc_x", color="#1f77b4")
        self.acc_y_line, = self.ax_axes.plot([], [], label="acc_y", color="#ff7f0e")
        self.acc_z_line, = self.ax_axes.plot([], [], label="acc_z", color="#2ca02c")
        self.gyro_x_line, = self.ax_axes.plot([], [], "--", label="gyro_x", color="#9467bd")
        self.gyro_y_line, = self.ax_axes.plot([], [], "--", label="gyro_y", color="#8c564b")
        self.gyro_z_line, = self.ax_axes.plot([], [], "--", label="gyro_z", color="#e377c2")
        self.ax_axes.set_title("Axis-Level Signals")
        self.ax_axes.set_xlabel("Sample")
        self.ax_axes.set_ylabel("Value")
        self.ax_axes.grid(alpha=0.3)
        self.ax_axes.legend(loc="upper right", fontsize=8, ncol=2)

        labels = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
        self.bar_container = self.ax_probs.bar(labels, np.zeros(len(CLASS_NAMES)), color="tab:green")
        self.ax_probs.set_title("Class Probabilities")
        self.ax_probs.set_ylim(0.0, 1.0)
        self.ax_probs.set_ylabel("Probability")
        self.ax_probs.set_xlabel("Class")
        self.ax_probs.grid(axis="y", alpha=0.3)
        self.prediction_text = self.ax_probs.text(
            0.01, 0.95, self.latest_pred_text, transform=self.ax_probs.transAxes, va="top", ha="left"
        )

        self.fall_line, = self.ax_fall.plot([], [], color="crimson", label="P(FALL)")
        self.pred_idx_line, = self.ax_fall.plot([], [], color="tab:gray", alpha=0.6, label="Pred Class ID")
        self.ax_fall.set_title("Fall Risk Trend")
        self.ax_fall.set_xlabel("Sample")
        self.ax_fall.set_ylabel("Probability / Class ID")
        self.ax_fall.set_ylim(0.0, max(1.0, len(CLASS_NAMES) - 1))
        self.ax_fall.grid(alpha=0.3)
        self.ax_fall.legend(loc="upper right")

        self.fig.tight_layout()
        plt.show(block=False)

    def add_sensor_sample(self, packet):
        acc_mag = float(np.linalg.norm(packet[:3]))
        gyro_mag = float(np.linalg.norm(packet[3:]))
        self.x_data.append(self.sample_idx)
        self.acc_x.append(packet[0])
        self.acc_y.append(packet[1])
        self.acc_z.append(packet[2])
        self.gyro_x.append(packet[3])
        self.gyro_y.append(packet[4])
        self.gyro_z.append(packet[5])
        self.acc_mag_data.append(acc_mag)
        self.gyro_mag_data.append(gyro_mag)
        self.sample_idx += 1

    def update_prediction(self, probs, label):
        self.latest_pred_text = f"Prediction: {label}"
        self.fall_prob_data.append(float(probs[4]))
        self.pred_class_data.append(float(np.argmax(probs)))

        for bar, p in zip(self.bar_container, probs):
            bar.set_height(float(p))
            bar.set_color("crimson" if label == "FALL" else "tab:green")

    def refresh(self):
        if self.x_data:
            x = list(self.x_data)
            self.acc_line.set_data(x, list(self.acc_mag_data))
            self.gyro_line.set_data(x, list(self.gyro_mag_data))
            self.ax_sensor.set_xlim(x[0], max(x[-1], x[0] + 1))

            y_max = max(max(self.acc_mag_data), max(self.gyro_mag_data))
            self.ax_sensor.set_ylim(0.0, max(1.0, y_max * 1.15))

            self.acc_x_line.set_data(x, list(self.acc_x))
            self.acc_y_line.set_data(x, list(self.acc_y))
            self.acc_z_line.set_data(x, list(self.acc_z))
            self.gyro_x_line.set_data(x, list(self.gyro_x))
            self.gyro_y_line.set_data(x, list(self.gyro_y))
            self.gyro_z_line.set_data(x, list(self.gyro_z))
            self.ax_axes.set_xlim(x[0], max(x[-1], x[0] + 1))
            axis_min = min(
                min(self.acc_x), min(self.acc_y), min(self.acc_z), min(self.gyro_x), min(self.gyro_y), min(self.gyro_z)
            )
            axis_max = max(
                max(self.acc_x), max(self.acc_y), max(self.acc_z), max(self.gyro_x), max(self.gyro_y), max(self.gyro_z)
            )
            pad = max(0.5, (axis_max - axis_min) * 0.1)
            self.ax_axes.set_ylim(axis_min - pad, axis_max + pad)

            if self.fall_prob_data:
                fall_x = x[-len(self.fall_prob_data) :]
                self.fall_line.set_data(fall_x, list(self.fall_prob_data))
                self.pred_idx_line.set_data(fall_x, list(self.pred_class_data))
                self.ax_fall.set_xlim(x[0], max(x[-1], x[0] + 1))

        self.prediction_text.set_text(self.latest_pred_text)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


def parse_and_predict(client, model, plotter):
    input_sensor = []
    token = ""
    packet = []
    packet_counter = 0

    while True:
        content = client.recv(1)
        if len(content) == 0:
            break

        char = content.decode("utf-8")
        if char == "!":
            packet = []
            token = ""
        elif char == ",":
            try:
                packet.append(float(token))
            except ValueError:
                packet = []
            token = ""
        elif char == "@":
            try:
                packet.append(float(token))
            except ValueError:
                packet = []
            token = ""

            if len(packet) != SENSOR_NUM:
                packet = []
                continue

            input_sensor.append(packet)
            plotter.add_sensor_sample(packet)
            if len(input_sensor) > STEP_SIZE:
                input_sensor.pop(0)

            if len(input_sensor) == STEP_SIZE:
                window = np.array(input_sensor).reshape(-1, STEP_SIZE, SENSOR_NUM)
                pred = model.predict(window, verbose=0)
                probs = pred[0]
                result = int(np.argmax(pred, axis=1)[0])
                label = CLASS_NAMES[result]
                print("prediction:", label)
                plotter.update_prediction(probs, label)

            packet_counter += 1
            if packet_counter % 3 == 0:
                plotter.refresh()

            packet = []
        else:
            token += char


def load_inference_model():
    if os.path.isfile(MODEL_FILE):
        return tf.keras.models.load_model(MODEL_FILE)
    try:
        return tf.keras.models.load_model(MODEL_DIR)
    except ValueError:
        try:
            layer = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
            inputs = tf.keras.Input(shape=(STEP_SIZE, SENSOR_NUM), dtype=tf.float32)
            outputs = layer(inputs)
            if isinstance(outputs, dict):
                outputs = next(iter(outputs.values()))
            return tf.keras.Model(inputs, outputs)
        except Exception as exc:
            raise SystemExit(
                "Could not load model from './model_x' or './model_x.keras'. "
                "Run Model_Training.py first to generate './model_x.keras'.\n"
                f"Original error: {exc}"
            )


def main():
    model = load_inference_model()
    model.summary()
    plotter = LivePlotter()

    server = socket.socket()
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Listening on {HOST}:{PORT}")
    record = input("Press R to start recording... ")
    if record.lower() != "r":
        server.close()
        return

    client, addr = server.accept()
    print("Connected from:", addr)

    try:
        parse_and_predict(client, model, plotter)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        server.close()
        plotter.close()


if __name__ == "__main__":
    main()
