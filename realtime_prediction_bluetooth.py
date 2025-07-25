"""
Echtzeit Tischtennisschlag-Erkennung über Bluetooth (Arduino Nicla)
Verwendet trainiertes PyTorch-Modell für die Vorhersage
Basiert auf arduino_ble_recorder.py für Bluetooth-Verbindung
"""

import asyncio
import time
import json
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from bleak import BleakScanner, BleakClient

# PyTorch CNN - identisch wie im Training
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


class RealtimeBluetoothPredictor:
    def __init__(self, 
                 model_path='models/best_cnn_pytorch.pth',
                 normalization_path='processed_data/normalization_minimal.json',
                 info_path='processed_data/info_minimal.json'):
        
        # Bluetooth-Konstanten
        self.UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
        self.UART_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
        
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
        
        # PyTorch Modell laden
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Tischtennis1DCNN(input_channels=self.num_features, num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Bluetooth client
        self.ble_client = None
        
        # Daten-Buffer
        self.data_buffer = deque(maxlen=self.window_size)
        self.raw_data_buffer = deque(maxlen=self.window_size)
        self.first_timestamp = None
        
        # Kalibrierung
        self.calibration_buffer = deque(maxlen=200)
        self.is_calibrated = False
        self.gyro_offset = np.zeros(3)
        
        # Bewegungserkennung
        self.movement_history = deque(maxlen=100)
        
        # Vorhersage-Steuerung
        self.last_prediction_time = 0
        self.cooldown_time = 2.0  # 2 Sekunden zwischen Vorhersagen
        self.confidence_threshold = 0.7
        
        print(f"Bluetooth Predictor initialisiert!")
        print(f"Modell: {self.class_names}")
        print(f"Window Size: {self.window_size}, Features: {len(self.features)}")

    async def find_nicla_device(self):
        """Sucht Arduino Nicla Gerät"""
        print("Suche Arduino Nicla...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and "nicla" in device.name.lower():
                print(f"Gefunden: {device.name} ({device.address})")
                return device
        
        print("Kein Nicla Gerät gefunden!")
        return None

    async def connect_to_device(self):
        """Verbindet mit Arduino Nicla"""
        device = await self.find_nicla_device()
        if not device:
            return False
        
        print(f"Verbinde mit {device.name}...")
        self.ble_client = BleakClient(device.address)
        
        try:
            await self.ble_client.connect()
            print("Verbunden!")
            return True
        except Exception as e:
            print(f"Verbindung fehlgeschlagen: {e}")
            return False

    def handle_bluetooth_data(self, sender, data):
        """Verarbeitet eingehende Bluetooth-Daten"""
        try:
            # Bytes zu Text konvertieren
            text = data.decode('utf-8').strip()
            
            if not text:
                return
            
            # Multiple Zeilen in einem Paket handhaben
            lines = text.split('\n')
            for line in lines:
                if not line or line.startswith("Timestamp"):
                    continue
                
                # CSV-Daten aufteilen
                values = line.split(',')
                if len(values) == 18:  # Erwartete Anzahl Werte
                    try:
                        # Timestamp verarbeiten
                        original_timestamp = int(values[0])
                        if self.first_timestamp is None:
                            self.first_timestamp = original_timestamp
                        
                        relative_timestamp = (original_timestamp - self.first_timestamp) / 1000.0
                        
                        # Sensordaten extrahieren
                        sensor_data = {
                            'timestamp': relative_timestamp,
                            'gyro_x': float(values[1]),
                            'gyro_y': float(values[2]),
                            'gyro_z': float(values[3]),
                            'acc_x': float(values[4]),
                            'acc_y': float(values[5]),
                            'acc_z': float(values[6]),
                            'mag_x': float(values[7]),
                            'mag_y': float(values[8]),
                            'mag_z': float(values[9]),
                            'bar': float(values[10]),
                            'quat_w': float(values[11]),
                            'quat_x': float(values[12]),
                            'quat_y': float(values[13]),
                            'quat_z': float(values[14]),
                            'lin_acc_x': float(values[15]),
                            'lin_acc_y': float(values[16]),
                            'lin_acc_z': float(values[17])
                        }
                        
                        self.process_sensor_data(sensor_data)
                        
                    except (ValueError, IndexError):
                        # Fehlerhafte Daten überspringen
                        continue
                        
        except Exception as e:
            # Korrupte Pakete überspringen
            pass

    def process_sensor_data(self, sensor_data):
        """Verarbeitet Sensordaten für Kalibrierung und Vorhersage"""
        # Rohdaten speichern
        self.raw_data_buffer.append({
            'gyro': [sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']],
            'lin_acc': [sensor_data['lin_acc_x'], sensor_data['lin_acc_y'], sensor_data['lin_acc_z']]
        })
        
        # Kalibrierung
        if not self.is_calibrated:
            self.calibration_buffer.append({
                'gyro': np.array([sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']])
            })
            
            if len(self.calibration_buffer) >= 200:
                self.calibrate()
        else:
            # Kalibrierte Gyro-Daten
            gyro_cal = np.array([sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']]) - self.gyro_offset
            
            # Feature-Array erstellen (nur die Trainings-Features)
            feature_vector = [
                gyro_cal[0], gyro_cal[1], gyro_cal[2],  # gyro_x, gyro_y, gyro_z
                sensor_data['lin_acc_x'], sensor_data['lin_acc_y'], sensor_data['lin_acc_z'],  # lin_acc
                sensor_data['quat_w'], sensor_data['quat_x'], sensor_data['quat_y'], sensor_data['quat_z']  # quaternion
            ]
            
            self.data_buffer.append(feature_vector)
            
            # Bewegungserkennung
            lin_acc_mag = np.sqrt(sensor_data['lin_acc_x']**2 + sensor_data['lin_acc_y']**2 + sensor_data['lin_acc_z']**2)
            gyro_mag = np.sqrt(gyro_cal[0]**2 + gyro_cal[1]**2 + gyro_cal[2]**2)
            
            self.movement_history.append({
                'lin_acc_mag': lin_acc_mag,
                'gyro_mag': gyro_mag,
                'timestamp': sensor_data['timestamp']
            })
            
            # Vorhersage-Logik
            self.check_for_prediction()

    def calibrate(self):
        """Kalibriert Gyro-Sensoren"""
        if len(self.calibration_buffer) < 100:
            return
        
        print("\nKalibriere Sensoren...")
        gyro_data = np.array([d['gyro'] for d in self.calibration_buffer])
        self.gyro_offset = np.mean(gyro_data, axis=0)
        
        print(f"Gyro-Offset: X={self.gyro_offset[0]:.2f}, Y={self.gyro_offset[1]:.2f}, Z={self.gyro_offset[2]:.2f}")
        self.is_calibrated = True
        self.calibration_buffer.clear()
        print("Kalibrierung abgeschlossen!\n")

    def detect_movement(self):
        """Erkennt Bewegung für Schlag-Trigger"""
        if len(self.movement_history) < 50:
            return False, 0
        
        recent = list(self.movement_history)[-50:]
        lin_acc_mags = np.array([d['lin_acc_mag'] for d in recent])
        gyro_mags = np.array([d['gyro_mag'] for d in recent])
        
        lin_acc_max = np.max(lin_acc_mags)
        lin_acc_std = np.std(lin_acc_mags)
        gyro_max = np.max(gyro_mags)
        gyro_std = np.std(gyro_mags)
        
        # Bewegung erkannt wenn starke Beschleunigung ODER starke Rotation
        movement_detected = (
            (lin_acc_max > 1.0 or lin_acc_std > 0.5) and
            (gyro_max > 50.0 or gyro_std > 20.0)
        )
        
        movement_score = (lin_acc_max) + (gyro_max / 100.0)
        
        return movement_detected, movement_score

    def check_for_prediction(self):
        """Prüft ob Vorhersage gemacht werden soll"""
        current_time = time.time()
        
        # Cooldown prüfen
        if current_time - self.last_prediction_time < self.cooldown_time:
            return
        
        # Genug Daten im Buffer?
        if len(self.data_buffer) < self.window_size:
            return
        
        # Bewegung erkannt?
        movement_detected, movement_score = self.detect_movement()
        
        if movement_detected and movement_score > 2.0:
            self.make_prediction()
            self.last_prediction_time = current_time

    def make_prediction(self):
        """Macht Schlag-Vorhersage"""
        if len(self.data_buffer) < self.window_size:
            return
        
        # Daten für Vorhersage vorbereiten
        window_data = np.array(list(self.data_buffer))  # (window_size, num_features)
        
        # Normalisieren
        window_data_norm = (window_data - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Für PyTorch umformen: (1, num_features, window_size)
        X = torch.tensor(window_data_norm.T[np.newaxis, :, :], dtype=torch.float32).to(self.device)
        
        # Vorhersage
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_value = confidence.item()
        
        # Nur bei hoher Konfidenz ausgeben
        if confidence_value >= self.confidence_threshold:
            stroke_name = self.class_names[predicted_class]
            print(f"🏓 SCHLAG ERKANNT: {stroke_name.upper()} (Konfidenz: {confidence_value:.2f})")
        else:
            print(f"🤔 Unklarer Schlag (Konfidenz: {confidence_value:.2f})")

    async def start_realtime_prediction(self):
        """Startet Echtzeit-Vorhersage"""
        print("\n=== Echtzeit Tischtennisschlag-Erkennung (Bluetooth) ===")
        
        # Mit Gerät verbinden
        if not await self.connect_to_device():
            print("Fehler: Kann nicht mit Gerät verbinden!")
            return
        
        print("Bitte halten Sie den Sensor für 2 Sekunden ruhig zur Kalibrierung...")
        
        # Datenempfang starten
        handler = lambda sender, data: self.handle_bluetooth_data(sender, data)
        await self.ble_client.start_notify(self.UART_TX_CHAR, handler)
        
        # Warten auf Kalibrierung
        while not self.is_calibrated:
            await asyncio.sleep(0.1)
        
        print("Bereit für Vorhersagen! Führen Sie Tischtennisschläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        
        try:
            # Haupt-Loop
            last_status_time = time.time()
            
            while True:
                await asyncio.sleep(0.1)
                
                # Status alle 10 Sekunden
                current_time = time.time()
                if current_time - last_status_time > 10:
                    buffer_fill = len(self.data_buffer)
                    movement_detected, movement_score = self.detect_movement()
                    print(f"Status: Buffer {buffer_fill}/{self.window_size}, Movement: {movement_score:.1f}")
                    last_status_time = current_time
                
        except KeyboardInterrupt:
            print("\nBeende Vorhersage...")
        
        finally:
            # Aufräumen
            if self.ble_client and self.ble_client.is_connected:
                await self.ble_client.stop_notify(self.UART_TX_CHAR)
                await self.ble_client.disconnect()
                print("Bluetooth-Verbindung getrennt")

    async def disconnect(self):
        """Trennt Bluetooth-Verbindung"""
        if self.ble_client and self.ble_client.is_connected:
            await self.ble_client.disconnect()


async def main():
    """Hauptfunktion"""
    print("Tischtennisschlag Echtzeit-Klassifikation über Bluetooth\n")
    
    try:
        predictor = RealtimeBluetoothPredictor(
            model_path='models/best_cnn_pytorch.pth',
            normalization_path='processed_data/normalization_minimal.json',
            info_path='processed_data/info_minimal.json'
        )
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        print("Stellen Sie sicher, dass die Modell- und Konfigurationsdateien existieren:")
        print("- models/best_cnn_pytorch.pth")
        print("- processed_data/normalization_minimal.json") 
        print("- processed_data/info_minimal.json")
        return
    
    try:
        await predictor.start_realtime_prediction()
    except Exception as e:
        print(f"\nFehler: {e}")
    finally:
        await predictor.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
