"""
Echtzeit Tischtennisschlag-Erkennung über WiFi (PyTorch Version)
Verwendet trainiertes PyTorch-Modell für die Vorhersage
"""

import numpy as np
import torch
import torch.nn as nn
import socket
import time
import threading
import queue
import json
from collections import deque
import osc_decoder

# PyTorch CNN muss wie im Training definiert sein
class Tischtennis1DCNN(nn.Module):
    def __init__(self, input_channels=10, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Kombinierte Klasse: Datenempfang, Kalibrierung, PyTorch-Vorhersage
class RealtimeWiFiPyTorchPredictor:
    def __init__(self, model_path='models/best_cnn_pytorch.pth', normalization_path='processed_data/normalization_minimal.json', info_path='processed_data/info_minimal.json', ports=[8001, 8010, 8011, 8012]):
        # Normalisierung laden
        with open(normalization_path, 'r') as f:
            norm_params = json.load(f)
        self.feature_means = np.array(norm_params['means'])
        self.feature_stds = np.array(norm_params['stds'])
        self.features = norm_params['features']
        # Info laden
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.class_names = info['stroke_types']
        self.window_size = info['window_size']
        self.num_features = len(self.features)
        # Modell laden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Tischtennis1DCNN(input_channels=self.num_features, num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        # Netzwerk
        self.ports = ports
        self.receive_sockets = []
        # Buffer
        self.data_buffer = deque(maxlen=self.window_size)
        self.raw_data_buffer = deque(maxlen=self.window_size)
        self.prediction_queue = queue.Queue()
        # Temporäre Sensor-Daten
        self.current_sensors_data = {}
        self.current_quaternion_data = {}
        self.current_linacc_data = {}
        # Kalibrierung
        self.calibration_buffer = deque(maxlen=200)
        self.is_calibrated = False
        self.gyro_offset = np.zeros(3)
        # Bewegungserkennung
        self.movement_history = deque(maxlen=100)
        # Threading
        self.running = False
        self.data_thread = None
        self.predict_thread = None
        print(f"PyTorch Modell geladen! Window Size: {self.window_size}, Features: {self.features}")

    def setup_sockets(self):
        self.receive_sockets = []
        for port in self.ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", port))
            sock.setblocking(False)
            self.receive_sockets.append(sock)
        print(f"Socket-Verbindungen erstellt auf Ports: {self.ports}")

    def receive_data(self):
        while self.running:
            for udp_socket in self.receive_sockets:
                try:
                    data, addr = udp_socket.recvfrom(2048)
                except socket.error:
                    continue
                else:
                    for message in osc_decoder.decode(data):
                        # message: [timestamp, address, ...floats]
                        if len(message) < 2:
                            continue
                        address = message[1]
                        floats = message[2:]
                        if address == '/sensors' and len(floats) >= 10:
                            self.current_sensors_data = {
                                'gyro_x': floats[0],
                                'gyro_y': floats[1],
                                'gyro_z': floats[2],
                                'acc_x': floats[3],
                                'acc_y': floats[4],
                                'acc_z': floats[5],
                                'mag_x': floats[6],
                                'mag_y': floats[7],
                                'mag_z': floats[8],
                                'baro': floats[9]
                            }
                        elif address == '/quaternion' and len(floats) >= 4:
                            self.current_quaternion_data = {
                                'w': floats[0],
                                'x': floats[1],
                                'y': floats[2],
                                'z': floats[3]
                            }
                        elif address == '/linear' and len(floats) >= 3:
                            self.current_linacc_data = {
                                'lin_acc_x': floats[0],
                                'lin_acc_y': floats[1],
                                'lin_acc_z': floats[2]
                            }
                            if self.current_sensors_data and self.current_quaternion_data and self.current_linacc_data:
                                self.process_complete_data()
            time.sleep(0.001)

    def process_complete_data(self):
        self.raw_data_buffer.append({
            'gyro': [self.current_sensors_data['gyro_x'], self.current_sensors_data['gyro_y'], self.current_sensors_data['gyro_z']],
            'lin_acc': [self.current_linacc_data['lin_acc_x'], self.current_linacc_data['lin_acc_y'], self.current_linacc_data['lin_acc_z']]
        })
        if not self.is_calibrated:
            self.calibration_buffer.append({
                'gyro': np.array([self.current_sensors_data['gyro_x'], self.current_sensors_data['gyro_y'], self.current_sensors_data['gyro_z']])
            })
            if len(self.calibration_buffer) >= 200:
                self.calibrate()
        else:
            gyro_cal = np.array([self.current_sensors_data['gyro_x'], self.current_sensors_data['gyro_y'], self.current_sensors_data['gyro_z']]) - self.gyro_offset
            data_point = [
                gyro_cal[0], gyro_cal[1], gyro_cal[2],
                self.current_linacc_data['lin_acc_x'],
                self.current_linacc_data['lin_acc_y'],
                self.current_linacc_data['lin_acc_z'],
                self.current_quaternion_data['w'],
                self.current_quaternion_data['x'],
                self.current_quaternion_data['y'],
                self.current_quaternion_data['z']
            ]
            self.data_buffer.append(data_point)
            lin_acc_magnitude = np.linalg.norm([
                self.current_linacc_data['lin_acc_x'],
                self.current_linacc_data['lin_acc_y'],
                self.current_linacc_data['lin_acc_z']
            ])
            gyro_magnitude = np.linalg.norm(gyro_cal)
            self.movement_history.append({
                'lin_acc_mag': lin_acc_magnitude,
                'gyro_mag': gyro_magnitude,
                'timestamp': time.time()
            })

    def calibrate(self):
        if len(self.calibration_buffer) < 100:
            print("Nicht genug Daten für Kalibrierung")
            return
        print("\nKalibriere Sensoren...")
        gyro_data = np.array([d['gyro'] for d in self.calibration_buffer])
        self.gyro_offset = np.mean(gyro_data, axis=0)
        print(f"Gyro-Offset: X={self.gyro_offset[0]:.2f}, Y={self.gyro_offset[1]:.2f}, Z={self.gyro_offset[2]:.2f}")
        self.is_calibrated = True
        self.calibration_buffer.clear()
        print("Kalibrierung abgeschlossen!\n")

    def detect_movement(self):
        if len(self.movement_history) < 50:
            return False, 0, 0
        recent = list(self.movement_history)[-50:]
        lin_acc_mags = np.array([d['lin_acc_mag'] for d in recent])
        gyro_mags = np.array([d['gyro_mag'] for d in recent])
        lin_acc_mean = np.mean(lin_acc_mags)
        lin_acc_std = np.std(lin_acc_mags)
        lin_acc_max = np.max(lin_acc_mags)
        gyro_mean = np.mean(gyro_mags)
        gyro_std = np.std(gyro_mags)
        gyro_max = np.max(gyro_mags)
        movement_detected = (
            (lin_acc_max > 1.0 or lin_acc_std > 0.5) and
            (gyro_max > 50.0 or gyro_std > 20.0)
        )
        movement_score = (lin_acc_max) + (gyro_max / 100.0)
        return movement_detected, movement_score, lin_acc_mean

    def predict_stroke(self):
        confidence_threshold = 0.7
        cooldown_time = 0.1         #modifiert: 2.0 Sekunde CPU Cooldown
        last_prediction_time = 0
        state = 'WAITING'
        movement_start_time = 0
        peak_movement_score = 0
        while self.running:
            if not self.is_calibrated:
                time.sleep(0.1)
                continue
            if len(self.data_buffer) == self.window_size:
                current_time = time.time()
                movement_detected, movement_score, lin_acc_mean = self.detect_movement()
                if state == 'WAITING':
                    if movement_detected:
                        state = 'MOVING'
                        movement_start_time = current_time
                        peak_movement_score = movement_score
                elif state == 'MOVING':
                    peak_movement_score = max(peak_movement_score, movement_score)
                    if not movement_detected:
                        movement_duration = current_time - movement_start_time
                        if movement_duration > 0.2 and peak_movement_score > 1.5:
                            state = 'PREDICTING'
                        else:
                            state = 'WAITING'
                elif state == 'PREDICTING':
                    if current_time - last_prediction_time > cooldown_time:
                        try:
                            t_pred_start = time.time()  # Startzeit der Vorhersage   
                            window_data = np.array(list(self.data_buffer))
                            window_scaled = (window_data - self.feature_means) / self.feature_stds
                            X = torch.tensor(window_scaled.T, dtype=torch.float32).unsqueeze(0).to(self.device)
                            with torch.no_grad():
                                logits = self.model(X)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            t_pred_end = time.time()    # Endzeit der Vorhersage
                            pred_time = (t_pred_end - t_pred_start) * 1000  # ms
                            predicted_class = np.argmax(probs)
                            confidence = probs[predicted_class]
                            if confidence > confidence_threshold:
                                result = {
                                    'class': self.class_names[predicted_class],
                                    'confidence': float(confidence),
                                    'timestamp': current_time,
                                    'all_probs': probs,
                                    'movement_score': peak_movement_score,
                                    'lin_acc_mean': lin_acc_mean,
                                    'prediction_time_ms': pred_time     # Vorhersagezeit in ms
                                }
                                print(f"[DEBUG] Vorhersagezeit: {pred_time:.2f} ms")    # Debug-Ausgabe
                                self.prediction_queue.put(result)
                                last_prediction_time = current_time
                        except Exception as e:
                            print(f"Vorhersagefehler: {e}")
                    state = 'COOLDOWN'
                elif state == 'COOLDOWN':
                    if current_time - last_prediction_time > cooldown_time:
                        state = 'WAITING'
            time.sleep(0.01)

    def start_data_reception(self):
        self.running = True
        self.data_thread = threading.Thread(target=self.receive_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        self.predict_thread = threading.Thread(target=self.predict_stroke)
        self.predict_thread.daemon = True
        self.predict_thread.start()
        print("Datenempfang und Vorhersage-Threads gestartet...")

    def stop_data_reception(self):
        self.running = False
        if self.data_thread:
            self.data_thread.join()
        if self.predict_thread:
            self.predict_thread.join()
        for sock in self.receive_sockets:
            sock.close()

    def start_realtime_prediction(self):
        print("\n=== Echtzeit Tischtennisschlag-Erkennung (WiFi, PyTorch) ===")
        print("Bitte halten Sie den Sensor für 2 Sekunden ruhig zur Kalibrierung...")
        while not self.is_calibrated:
            time.sleep(0.1)
        print("Bereit für Vorhersagen! Führen Sie Tischtennisschläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        last_status_time = time.time()
        try:
            while True:
                current_time = time.time()
                if not self.prediction_queue.empty():
                    result = self.prediction_queue.get()
                    print(f"\n{'='*60}")
                    print(f"⚡ SCHLAG ERKANNT!")
                    print(f"   Typ: {result['class']}")
                    print(f"   Konfidenz: {result['confidence']:.1%}")
                    print(f"   Bewegungsintensität: {result['movement_score']:.2f}")
                    print(f"   Durchschn. Lin. Beschleunigung: {result['lin_acc_mean']:.1f} g")
                    print(f"   Vorhersagezeit: {result.get('prediction_time_ms', 0):.2f} ms")
                    print(f"\n   Wahrscheinlichkeiten:")
                    for i, (name, prob) in enumerate(zip(self.class_names, result['all_probs'])):
                        bar = '█' * int(prob * 20)
                        print(f"   {name:20} {bar:20} {prob:.1%}")
                    print(f"{'='*60}\n")
                if current_time - last_status_time > 1:  #modifiert von> 3 Sekunden Status-Update
                    if len(self.movement_history) > 0:
                        recent = list(self.movement_history)[-10:]
                        current_lin_acc = np.mean([d['lin_acc_mag'] for d in recent])
                        current_gyro = np.mean([d['gyro_mag'] for d in recent])
                        buffer_fill = len(self.data_buffer) / self.window_size * 100
                        status = f"Buffer: {buffer_fill:.0f}% | "
                        if self.is_calibrated:
                            status += f"Lin Acc: {current_lin_acc:.1f}g | Gyro: {current_gyro:.0f}°/s"
                        else:
                            status += "Kalibriere..."
                        print(f"\r{status} | Warte auf Schlag...", end='', flush=True)
                    last_status_time = current_time
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n\nBeende Vorhersage...")
        finally:
            self.stop_data_reception()

def main():
    print("Tischtennisschlag Echtzeit-Klassifikation über WiFi (PyTorch)\n")
    try:
        predictor = RealtimeWiFiPyTorchPredictor(
            model_path='models/best_cnn_pytorch.pth',
            normalization_path='processed_data/normalization_minimal.json',
            info_path='processed_data/info_minimal.json',
            ports=[8001, 8010, 8011, 8012]
        )
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return
    predictor.setup_sockets()
    predictor.start_data_reception()
    try:
        predictor.start_realtime_prediction()
    except Exception as e:
        print(f"\nFehler: {e}")
    finally:
        predictor.stop_data_reception()

if __name__ == "__main__":
    main()
