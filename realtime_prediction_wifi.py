"""
Echtzeit-Vorhersage für Tischtennisschläge über WiFi/OSC
Verwendet das trainierte Modell zur Live-Klassifikation mit NGIMU
Verbesserte Version mit Kalibrierung und Ruhezustandserkennung
"""

import numpy as np
import tensorflow as tf
from pythonosc import dispatcher, osc_server
import pickle
import time
from collections import deque
import threading
import queue

class RealtimeWiFiPredictor:
    def __init__(self, model_path, scaler_path, ip="0.0.0.0", port=8001):
        # Modell laden
        print("Lade Modell...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Scaler laden
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Modell-Info laden (falls vorhanden)
        try:
            with open('models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
                self.class_names = self.model_info['class_names']
        except:
            self.class_names = ['Vorhand Topspin', 'Vorhand Schupf', 'Rückhand Topspin', 'Rückhand Schupf']
        
        # OSC/WiFi Einstellungen
        self.ip = ip
        self.port = port
        self.dispatcher = dispatcher.Dispatcher()
        self.server = None
        
        # Daten-Buffer
        self.window_size = 200  # 2 Sekunden @ 100Hz
        self.data_buffer = deque(maxlen=self.window_size)
        self.raw_data_buffer = deque(maxlen=self.window_size)  # Für Debug
        self.prediction_queue = queue.Queue()
        
        # Kalibrierungs-Buffer
        self.calibration_buffer = deque(maxlen=200)  # 2 Sekunden für Kalibrierung
        self.is_calibrated = False
        self.acc_offset = np.array([0.0, 0.0, 0.0])
        self.gyro_offset = np.array([0.0, 0.0, 0.0])
        
        # Bewegungserkennung
        self.movement_history = deque(maxlen=100)  # 1 Sekunde Historie
        
        # Aktuelle Quaternion-Daten
        self.current_quaternion = {'w': 0, 'x': 0, 'y': 0, 'z': 0}
        
        # Threading
        self.running = False
        self.server_thread = None
        self.predict_thread = None
        
        # Setup OSC handlers
        self.setup_handlers()
        
    def setup_handlers(self):
        """OSC-Handler für NGIMU-Daten einrichten"""
        self.dispatcher.map("/sensors", self.handle_sensors)
        self.dispatcher.map("/quaternion", self.handle_quaternion)
        
    def handle_sensors(self, address, *args):
        """Verarbeitet Sensor-Daten (Gyro, Acc, Mag)"""
        if len(args) >= 10:
            # Format: timestamp, gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ
            timestamp = args[0]
            gyro = np.array(args[1:4])
            acc = np.array(args[4:7])
            
            # Rohdaten für Debug speichern
            self.raw_data_buffer.append({
                'acc': acc.copy(),
                'gyro': gyro.copy()
            })
            
            # Kalibrierung falls noch nicht erfolgt
            if not self.is_calibrated:
                self.calibration_buffer.append({
                    'acc': acc,
                    'gyro': gyro
                })
                if len(self.calibration_buffer) >= 200:
                    self.calibrate()
            else:
                # Kalibrierte Werte
                acc_cal = acc - self.acc_offset
                gyro_cal = gyro - self.gyro_offset
                
                # Erstelle Datenpunkt (6 Werte wie im Training)
                data_point = [
                    acc_cal[0], acc_cal[1], acc_cal[2],
                    gyro_cal[0], gyro_cal[1], gyro_cal[2]
                ]
                
                self.data_buffer.append(data_point)
                
                # Bewegungsintensität für Erkennung
                acc_magnitude = np.linalg.norm(acc_cal)
                gyro_magnitude = np.linalg.norm(gyro_cal)
                
                # Speichere Bewegungshistorie
                self.movement_history.append({
                    'acc_mag': acc_magnitude,
                    'gyro_mag': gyro_magnitude,
                    'timestamp': time.time()
                })
    
    def calibrate(self):
        """Kalibriert die Sensoren basierend auf Ruhezustand"""
        if len(self.calibration_buffer) < 100:
            print("Nicht genug Daten für Kalibrierung")
            return
            
        print("\nKalibriere Sensoren...")
        
        # Berechne Durchschnittswerte im Ruhezustand
        acc_data = np.array([d['acc'] for d in self.calibration_buffer])
        gyro_data = np.array([d['gyro'] for d in self.calibration_buffer])
        
        # Gyro-Offset (sollte 0 sein im Ruhezustand)
        self.gyro_offset = np.mean(gyro_data, axis=0)
        
        # Acc-Offset (Z-Achse sollte ~9.81 sein, X/Y sollten 0 sein)
        acc_mean = np.mean(acc_data, axis=0)
        self.acc_offset = np.array([acc_mean[0], acc_mean[1], acc_mean[2] - 9.81])
        
        print(f"Acc-Offset: X={self.acc_offset[0]:.2f}, Y={self.acc_offset[1]:.2f}, Z={self.acc_offset[2]:.2f}")
        print(f"Gyro-Offset: X={self.gyro_offset[0]:.2f}, Y={self.gyro_offset[1]:.2f}, Z={self.gyro_offset[2]:.2f}")
        
        # Überprüfe ob Kalibrierung sinnvoll ist
        if abs(acc_mean[2]) > 20:  # Wenn Z-Acc > 20g, stimmt was nicht
            print("WARNUNG: Ungewöhnlich hohe Beschleunigungswerte!")
            print("Überprüfen Sie die Sensorausrichtung und Einheiten")
        
        self.is_calibrated = True
        self.calibration_buffer.clear()
        print("Kalibrierung abgeschlossen!\n")
    
    def handle_quaternion(self, address, *args):
        """Verarbeitet Quaternion-Daten"""
        if len(args) >= 5:
            self.current_quaternion = {
                'w': args[1],
                'x': args[2],
                'y': args[3],
                'z': args[4]
            }
    
    def detect_movement(self):
        """Verbesserte Bewegungserkennung mit mehreren Kriterien"""
        if len(self.movement_history) < 50:
            return False, 0, 0
            
        # Analysiere die letzten 0.5 Sekunden
        recent = list(self.movement_history)[-50:]
        
        # Berechne Statistiken
        acc_mags = np.array([d['acc_mag'] for d in recent])
        gyro_mags = np.array([d['gyro_mag'] for d in recent])
        
        # Kriterium 1: Beschleunigung
        acc_mean = np.mean(acc_mags)
        acc_std = np.std(acc_mags)
        acc_max = np.max(acc_mags)
        
        # Kriterium 2: Winkelgeschwindigkeit
        gyro_mean = np.mean(gyro_mags)
        gyro_std = np.std(gyro_mags)
        gyro_max = np.max(gyro_mags)
        
        # Ruhezustand: Acc ≈ 9.81g (Gravitation), Gyro ≈ 0
        # Bewegung erkannt wenn:
        # 1. Acc deutlich von Gravitation abweicht
        # 2. Gyro zeigt Rotation
        # 3. Hohe Variabilität in den Daten
        
        acc_deviation = abs(acc_mean - 9.81)
        
        movement_detected = (
            (acc_deviation > 1.0 or acc_std > 1.0 or acc_max > 12.0) and
            (gyro_max > 50.0 or gyro_std > 20.0)
        )
        
        # Berechne Bewegungsintensität
        movement_score = (acc_deviation / 9.81) + (gyro_max / 100.0)
        
        return movement_detected, movement_score, acc_mean
    
    def predict_stroke(self):
        """Thread: Führt Vorhersagen durch wenn genug Daten vorhanden"""
        confidence_threshold = 0.7  # Hoher Schwellwert
        cooldown_time = 2.0  # 2 Sekunden zwischen Vorhersagen
        last_prediction_time = 0
        
        # State Machine
        state = 'WAITING'  # WAITING -> MOVING -> PREDICTING -> COOLDOWN
        movement_start_time = 0
        peak_movement_score = 0
        
        while self.running:
            if not self.is_calibrated:
                time.sleep(0.1)
                continue
                
            if len(self.data_buffer) == self.window_size:
                current_time = time.time()
                
                # Bewegungserkennung
                movement_detected, movement_score, acc_mean = self.detect_movement()
                
                # State Machine Logic
                if state == 'WAITING':
                    if movement_detected:
                        state = 'MOVING'
                        movement_start_time = current_time
                        peak_movement_score = movement_score
                        
                elif state == 'MOVING':
                    peak_movement_score = max(peak_movement_score, movement_score)
                    
                    if not movement_detected:
                        # Bewegung beendet
                        movement_duration = current_time - movement_start_time
                        
                        if movement_duration > 0.2 and peak_movement_score > 1.5:
                            state = 'PREDICTING'
                        else:
                            state = 'WAITING'
                            
                elif state == 'PREDICTING':
                    if current_time - last_prediction_time > cooldown_time:
                        # Vorhersage durchführen
                        try:
                            window_data = np.array(list(self.data_buffer))
                            
                            # Debug: Zeige Datenbereich
                            if len(self.raw_data_buffer) > 0:
                                recent_raw = list(self.raw_data_buffer)[-10:]
                                raw_acc = np.array([d['acc'] for d in recent_raw])
                                print(f"\nDebug - Rohe Acc-Werte (letzte 10):")
                                print(f"  X: {np.mean(raw_acc[:, 0]):.1f} ± {np.std(raw_acc[:, 0]):.1f}")
                                print(f"  Y: {np.mean(raw_acc[:, 1]):.1f} ± {np.std(raw_acc[:, 1]):.1f}")
                                print(f"  Z: {np.mean(raw_acc[:, 2]):.1f} ± {np.std(raw_acc[:, 2]):.1f}")
                            
                            # Normalisierung
                            window_scaled = self.scaler.transform(window_data)
                            
                            # Vorhersage
                            X = window_scaled.reshape(1, self.window_size, 6)
                            prediction = self.model.predict(X, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = np.max(prediction[0])
                            
                            if confidence > confidence_threshold:
                                result = {
                                    'class': self.class_names[predicted_class],
                                    'confidence': confidence,
                                    'timestamp': current_time,
                                    'all_probs': prediction[0],
                                    'movement_score': peak_movement_score,
                                    'acc_mean': acc_mean
                                }
                                
                                self.prediction_queue.put(result)
                                last_prediction_time = current_time
                                
                        except Exception as e:
                            print(f"Vorhersagefehler: {e}")
                    
                    state = 'COOLDOWN'
                    
                elif state == 'COOLDOWN':
                    if current_time - last_prediction_time > cooldown_time:
                        state = 'WAITING'
            
            time.sleep(0.01)
    
    def start_server(self):
        """Startet den OSC-Server"""
        print(f"Starte OSC-Server auf {self.ip}:{self.port}")
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self.dispatcher
        )
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print("OSC-Server läuft. Warte auf NGIMU-Daten...")
        print(f"NGIMU sollte an 192.168.1.2:{self.port} senden")
    
    def stop_server(self):
        """Stoppt den OSC-Server"""
        if self.server:
            self.server.shutdown()
            if self.server_thread:
                self.server_thread.join()
    
    def check_connection(self, timeout=5):
        """Überprüft ob NGIMU Daten sendet"""
        print(f"\nÜberprüfe NGIMU-Verbindung für {timeout} Sekunden...")
        
        start_time = time.time()
        initial_buffer_size = len(self.raw_data_buffer)
        
        while time.time() - start_time < timeout:
            if len(self.raw_data_buffer) > initial_buffer_size:
                print("✓ NGIMU sendet Daten!")
                
                # Zeige Beispieldaten
                if len(self.raw_data_buffer) > 0:
                    sample = list(self.raw_data_buffer)[-1]
                    print(f"\nBeispiel-Rohdaten:")
                    print(f"  Acc: X={sample['acc'][0]:.1f}, Y={sample['acc'][1]:.1f}, Z={sample['acc'][2]:.1f}")
                    print(f"  Gyro: X={sample['gyro'][0]:.1f}, Y={sample['gyro'][1]:.1f}, Z={sample['gyro'][2]:.1f}")
                
                return True
            time.sleep(0.1)
        
        print("✗ Keine Daten von NGIMU empfangen!")
        return False
    
    def start_realtime_prediction(self):
        """Startet die Echtzeit-Vorhersage"""
        print("\n=== Echtzeit Tischtennisschlag-Erkennung (WiFi) ===")
        
        # Verbindung prüfen
        if not self.check_connection():
            print("\nBitte überprüfen Sie:")
            print("1. NGIMU ist eingeschaltet")
            print("2. Sie sind mit dem NGIMU WiFi verbunden")
            print("3. NGIMU Send Settings korrekt konfiguriert")
            return
        
        print("\nBitte halten Sie den Sensor für 2 Sekunden ruhig zur Kalibrierung...")
        
        # Warte auf Kalibrierung
        while not self.is_calibrated:
            time.sleep(0.1)
        
        print("Bereit für Vorhersagen!")
        print("Führen Sie Tischtennisschläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        
        self.running = True
        
        # Vorhersage-Thread starten
        self.predict_thread = threading.Thread(target=self.predict_stroke)
        self.predict_thread.start()
        
        # Status-Variablen
        last_status_time = time.time()
        
        # Vorhersagen anzeigen
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
                    print(f"   Durchschn. Beschleunigung: {result['acc_mean']:.1f} g")
                    
                    # Zeige alle Wahrscheinlichkeiten
                    print(f"\n   Wahrscheinlichkeiten:")
                    for i, (name, prob) in enumerate(zip(self.class_names, result['all_probs'])):
                        bar = '█' * int(prob * 20)
                        print(f"   {name:20} {bar:20} {prob:.1%}")
                    print(f"{'='*60}\n")
                
                # Status-Update alle 3 Sekunden
                if current_time - last_status_time > 3:
                    if len(self.movement_history) > 0:
                        recent = list(self.movement_history)[-10:]
                        current_acc = np.mean([d['acc_mag'] for d in recent])
                        current_gyro = np.mean([d['gyro_mag'] for d in recent])
                        buffer_fill = len(self.data_buffer) / self.window_size * 100
                        
                        status = f"Buffer: {buffer_fill:.0f}% | "
                        if self.is_calibrated:
                            status += f"Acc: {current_acc:.1f}g | Gyro: {current_gyro:.0f}°/s"
                        else:
                            status += "Kalibriere..."
                        
                        print(f"\r{status} | Warte auf Schlag...", end='', flush=True)
                    last_status_time = current_time
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nBeende Vorhersage...")
            self.running = False
            if self.predict_thread:
                self.predict_thread.join()
            self.stop_server()

def main():
    """Hauptfunktion"""
    print("Tischtennisschlag Echtzeit-Klassifikation über WiFi\n")
    
    # Prüfe ob Modell existiert
    try:
        predictor = RealtimeWiFiPredictor(
            model_path='models/best_model.h5',
            scaler_path='processed_data/scaler.pkl',
            ip="0.0.0.0",
            port=8001
        )
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        print("\nStellen Sie sicher, dass Sie zuerst trainiert haben:")
        print("1. python data_preprocessing.py")
        print("2. python train_1d_cnn.py")
        return
    
    # Server starten
    predictor.start_server()
    
    try:
        # Echtzeit-Vorhersage starten
        predictor.start_realtime_prediction()
    except Exception as e:
        print(f"\nFehler: {e}")
    finally:
        predictor.stop_server()

if __name__ == "__main__":
    main()
