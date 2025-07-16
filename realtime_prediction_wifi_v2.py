"""
Echtzeit-Vorhersage für Tischtennisschläge über WiFi/OSC - Optimierte Version
Schnellere Reaktionszeit für professionelle Spielgeschwindigkeit (0.5-1s zwischen Schlägen)
"""

import numpy as np
import tensorflow as tf
import pickle
import time
from collections import deque
import threading
import queue
import socket
import osc_decoder

class RealtimeWiFiPredictor:
    def __init__(self, model_path, scaler_path, ip="0.0.0.0", ports=[8001, 8010, 8011, 8012]):
        # Modell laden
        print("Lade Modell...")
        self.model = tf.keras.models.load_model(model_path)
        
        # Scaler laden
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Modell-Info laden
        try:
            with open('models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
                self.class_names = self.model_info['class_names']
                self.feature_names = self.model_info.get('feature_names', [])
                self.num_features = self.model_info.get('num_features', 10)
        except:
            self.class_names = ['Vorhand Topspin', 'Vorhand Schupf', 'Rückhand Topspin', 'Rückhand Schupf']
            self.feature_names = ['gyro_x', 'gyro_y', 'gyro_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
                                 'quat_w', 'quat_x', 'quat_y', 'quat_z']
            self.num_features = 10
        
        print(f"Modell verwendet {self.num_features} Features: {self.feature_names}")
        
        # Netzwerk-Einstellungen
        self.ip = ip
        self.ports = ports
        self.ngimu_ip = "192.168.1.1"
        self.ngimu_send_port = 9000
        
        # Socket-Verbindungen
        self.send_socket = None
        self.receive_sockets = []
        
        # Daten-Buffer - behalten bei 200 für gute Erkennung
        self.window_size = 200  # 2 Sekunden @ 100Hz
        self.data_buffer = deque(maxlen=self.window_size)
        self.raw_data_buffer = deque(maxlen=self.window_size)
        self.prediction_queue = queue.Queue()
        
        # Temporäre Datenspeicher
        self.current_sensors_data = {}
        self.current_quaternion_data = {}
        self.current_linacc_data = {}
        
        # Kalibrierungs-Buffer
        self.calibration_buffer = deque(maxlen=200)
        self.is_calibrated = False
        self.gyro_offset = np.array([0.0, 0.0, 0.0])
        
        # Bewegungserkennung - 0.5s wie ursprünglich
        self.movement_history = deque(maxlen=100)
        
        # Threading
        self.running = False
        self.data_thread = None
        self.predict_thread = None
        
    def setup_sockets(self):
        """Erstellt und konfiguriert die Socket-Verbindungen"""
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.receive_sockets = []
        for port in self.ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", port))
            sock.setblocking(False)
            self.receive_sockets.append(sock)
        
        self.send_socket.sendto(bytes("/identify\0\0\0,\0\0\0", "utf-8"), 
                                (self.ngimu_ip, self.ngimu_send_port))
        
        print(f"Socket-Verbindungen erstellt auf Ports: {self.ports}")
    
    def receive_data(self):
        """Empfängt Daten vom NGIMU (läuft in separatem Thread)"""
        while self.running:
            for udp_socket in self.receive_sockets:
                try:
                    data, addr = udp_socket.recvfrom(2048)
                except socket.error:
                    pass
                else:
                    for message in osc_decoder.decode(data):
                        if len(message) >= 2:
                            osc_address = message[1]
                            
                            if osc_address == '/sensors' and len(message) == 12:
                                self.current_sensors_data = {
                                    'gyro_x': message[2],
                                    'gyro_y': message[3],
                                    'gyro_z': message[4],
                                    'acc_x': message[5],
                                    'acc_y': message[6],
                                    'acc_z': message[7],
                                    'mag_x': message[8],
                                    'mag_y': message[9],
                                    'mag_z': message[10],
                                    'baro': message[11]
                                }
                                
                            elif osc_address == '/quaternion' and len(message) == 6:
                                self.current_quaternion_data = {
                                    'w': message[2],
                                    'x': message[3],
                                    'y': message[4],
                                    'z': message[5]
                                }
                                
                            elif osc_address == '/linear' and len(message) == 5:
                                self.current_linacc_data = {
                                    'lin_acc_x': message[2],
                                    'lin_acc_y': message[3],
                                    'lin_acc_z': message[4]
                                }
                                
                                if self.current_sensors_data and self.current_quaternion_data and self.current_linacc_data:
                                    self.process_complete_data()
            
            time.sleep(0.001)
    
    def process_complete_data(self):
        """Verarbeitet vollständige Datensätze"""
        self.raw_data_buffer.append({
            'gyro': [self.current_sensors_data['gyro_x'], 
                    self.current_sensors_data['gyro_y'], 
                    self.current_sensors_data['gyro_z']],
            'lin_acc': [self.current_linacc_data['lin_acc_x'],
                       self.current_linacc_data['lin_acc_y'],
                       self.current_linacc_data['lin_acc_z']]
        })
        
        if not self.is_calibrated:
            self.calibration_buffer.append({
                'gyro': np.array([self.current_sensors_data['gyro_x'],
                                 self.current_sensors_data['gyro_y'],
                                 self.current_sensors_data['gyro_z']])
            })
            if len(self.calibration_buffer) >= 200:
                self.calibrate()
        else:
            gyro_cal = np.array([self.current_sensors_data['gyro_x'],
                                self.current_sensors_data['gyro_y'],
                                self.current_sensors_data['gyro_z']]) - self.gyro_offset
            
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
        """Kalibriert die Sensoren basierend auf Ruhezustand"""
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
        """Bewegungserkennung mit bewährten Parametern"""
        if len(self.movement_history) < 50:
            return False, 0, 0
            
        # Analysiere die letzten 0.5 Sekunden
        recent = list(self.movement_history)[-50:]
        
        lin_acc_mags = np.array([d['lin_acc_mag'] for d in recent])
        gyro_mags = np.array([d['gyro_mag'] for d in recent])
        
        # Bewährte Kriterien
        lin_acc_mean = np.mean(lin_acc_mags)
        lin_acc_std = np.std(lin_acc_mags)
        lin_acc_max = np.max(lin_acc_mags)
        
        gyro_mean = np.mean(gyro_mags)
        gyro_std = np.std(gyro_mags)
        gyro_max = np.max(gyro_mags)
        
        # Bewegung erkannt wenn deutliche Abweichung vom Ruhezustand
        movement_detected = (
            #(lin_acc_max > 1.0 or lin_acc_std > 1) and
            #(gyro_max > 50.0 or gyro_std > 20.0)
            (lin_acc_std > 1.0) and
            ( gyro_std > 100.0)
        )
        
        movement_score = (lin_acc_max) + (gyro_max / 100.0)
        
        return movement_detected, movement_score, lin_acc_mean
    
    def predict_stroke(self):
        """Optimierter Thread für schnellere Vorhersagen"""
        confidence_threshold = 0.8
        cooldown_time = 0.1  # Reduziert von 2.0s auf 0.5s - Hauptoptimierung!
        last_prediction_time = 0
        
        # State Machine
        state = 'WAITING'
        movement_start_time = 0
        peak_movement_score = 0
        
        while self.running:
            if not self.is_calibrated:
                time.sleep(0.1)
                continue
                
            if len(self.data_buffer) == self.window_size:  # Voller Buffer wie gewünscht
                current_time = time.time()
                
                movement_detected, movement_score, lin_acc_mean = self.detect_movement()
                
                if state == 'WAITING':
                    if movement_detected and (current_time - last_prediction_time) > cooldown_time:
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
                    try:
                        window_data = np.array(list(self.data_buffer))
                        
                        # Debug Info
                        if len(self.raw_data_buffer) > 0:
                            recent_raw = list(self.raw_data_buffer)[-10:]
                            raw_lin_acc = np.array([d['lin_acc'] for d in recent_raw])
                            print(f"\nDebug - Lineare Acc-Werte (letzte 10):")
                            print(f"  X: {np.mean(raw_lin_acc[:, 0]):.1f} ± {np.std(raw_lin_acc[:, 0]):.1f}")
                            print(f"  Y: {np.mean(raw_lin_acc[:, 1]):.1f} ± {np.std(raw_lin_acc[:, 1]):.1f}")
                            print(f"  Z: {np.mean(raw_lin_acc[:, 2]):.1f} ± {np.std(raw_lin_acc[:, 2]):.1f}")
                        
                        # Normalisierung
                        window_scaled = self.scaler.transform(window_data)
                        
                        # Vorhersage
                        X = window_scaled.reshape(1, self.window_size, self.num_features)
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
                                'lin_acc_mean': lin_acc_mean,
                                'reaction_time': current_time - movement_start_time
                            }
                            
                            self.prediction_queue.put(result)
                            last_prediction_time = current_time
                            
                    except Exception as e:
                        print(f"Vorhersagefehler: {e}")
                    
                    state = 'WAITING'
            
            time.sleep(0.01)
    
    def start_data_reception(self):
        """Startet den Datenempfang in separaten Threads"""
        self.running = True
        
        self.data_thread = threading.Thread(target=self.receive_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        self.predict_thread = threading.Thread(target=self.predict_stroke)
        self.predict_thread.daemon = True
        self.predict_thread.start()
        
        print("Datenempfang und Vorhersage-Threads gestartet...")
    
    def stop_data_reception(self):
        """Stoppt alle Threads und schließt Sockets"""
        self.running = False
        
        if self.data_thread:
            self.data_thread.join()
        if self.predict_thread:
            self.predict_thread.join()
        
        for sock in self.receive_sockets:
            sock.close()
        if self.send_socket:
            self.send_socket.close()
    
    def check_connection(self, timeout=5):
        """Überprüft ob NGIMU Daten sendet"""
        print(f"\nÜberprüfe NGIMU-Verbindung für {timeout} Sekunden...")
        
        start_time = time.time()
        initial_buffer_size = len(self.raw_data_buffer)
        
        while time.time() - start_time < timeout:
            if len(self.raw_data_buffer) > initial_buffer_size:
                print("✓ NGIMU sendet Daten!")
                
                if len(self.raw_data_buffer) > 0:
                    sample = list(self.raw_data_buffer)[-1]
                    print(f"\nBeispiel-Rohdaten:")
                    print(f"  Gyro: X={sample['gyro'][0]:.1f}, Y={sample['gyro'][1]:.1f}, Z={sample['gyro'][2]:.1f}")
                    print(f"  Lin Acc: X={sample['lin_acc'][0]:.1f}, Y={sample['lin_acc'][1]:.1f}, Z={sample['lin_acc'][2]:.1f}")
                
                return True
            time.sleep(0.1)
        
        print("✗ Keine Daten von NGIMU empfangen!")
        return False
    
    def start_realtime_prediction(self):
        """Startet die Echtzeit-Vorhersage"""
        print("\n=== Echtzeit Tischtennisschlag-Erkennung V2 (Optimierte Version) ===")
        print(f"Verbinde mit NGIMU auf {self.ngimu_ip}...")
        
        if not self.check_connection():
            print("\nBitte überprüfen Sie:")
            print("1. NGIMU ist eingeschaltet")
            print("2. Sie sind mit dem NGIMU WiFi verbunden")
            print("3. NGIMU Send Settings korrekt konfiguriert")
            print("4. NGIMU sendet /sensors, /quaternion und /linear Nachrichten")
            return
        
        print("\nBitte halten Sie den Sensor für 2 Sekunden ruhig zur Kalibrierung...")
        
        while not self.is_calibrated:
            time.sleep(0.1)
        
        print("Bereit für Vorhersagen!")
        print("Cooldown-Zeit: 0.5 Sekunden")
        print("Führen Sie Tischtennisschläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        
        last_status_time = time.time()
        prediction_count = 0
        
        try:
            while True:
                current_time = time.time()
                
                if not self.prediction_queue.empty():
                    result = self.prediction_queue.get()
                    prediction_count += 1
                    
                    print(f"\n{'='*60}")
                    print(f"⚡ SCHLAG #{prediction_count} ERKANNT! (Reaktionszeit: {result['reaction_time']*1000:.0f}ms)")
                    print(f"   Typ: {result['class']}")
                    print(f"   Konfidenz: {result['confidence']:.1%}")
                    print(f"   Bewegungsintensität: {result['movement_score']:.2f}")
                    print(f"   Durchschn. Lin. Beschleunigung: {result['lin_acc_mean']:.1f} g")
                    
                    print(f"\n   Wahrscheinlichkeiten:")
                    for i, (name, prob) in enumerate(zip(self.class_names, result['all_probs'])):
                        bar = '█' * int(prob * 20)
                        print(f"   {name:20} {bar:20} {prob:.1%}")
                    print(f"{'='*60}\n")
                
                # Status-Update alle 3 Sekunden
                if current_time - last_status_time > 3:
                    if len(self.movement_history) > 0:
                        recent = list(self.movement_history)[-10:]
                        current_lin_acc = np.mean([d['lin_acc_mag'] for d in recent])
                        current_gyro = np.mean([d['gyro_mag'] for d in recent])
                        buffer_fill = len(self.data_buffer) / self.window_size * 100
                        
                        status = f"Buffer: {buffer_fill:.0f}% | "
                        if self.is_calibrated:
                            status += f"Lin Acc: {current_lin_acc:.1f}g | Gyro: {current_gyro:.0f}°/s | "
                            status += f"Schläge: {prediction_count}"
                        else:
                            status += "Kalibriere..."
                        
                        print(f"\r{status} | Warte auf Schlag...", end='', flush=True)
                    last_status_time = current_time
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print(f"\n\nBeende Vorhersage... (Insgesamt {prediction_count} Schläge erkannt)")
        finally:
            self.stop_data_reception()

def main():
    """Hauptfunktion"""
    print("Tischtennisschlag Echtzeit-Klassifikation V2 - Konservative Optimierung\n")
    
    try:
        predictor = RealtimeWiFiPredictor(
            model_path='models/best_model.h5',
            scaler_path='processed_data/scaler.pkl',
            ports=[8001, 8010, 8011, 8012]
        )
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        print("\nStellen Sie sicher, dass Sie zuerst trainiert haben:")
        print("1. python data_preprocessing.py")
        print("2. python train_1d_cnn.py")
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
