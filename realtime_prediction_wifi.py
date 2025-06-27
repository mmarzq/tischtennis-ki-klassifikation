"""
Echtzeit-Vorhersage für Tischtennisschläge über WiFi/OSC
Verwendet das trainierte Modell zur Live-Klassifikation mit NGIMU
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
        self.prediction_queue = queue.Queue()
        
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
            gyro = args[1:4]
            acc = args[4:7]
            
            # Erstelle Datenpunkt (6 Werte wie im Training)
            data_point = [
                acc[0], acc[1], acc[2],  # Acc X, Y, Z
                gyro[0], gyro[1], gyro[2]  # Gyro X, Y, Z
            ]
            
            self.data_buffer.append(data_point)
    
    def handle_quaternion(self, address, *args):
        """Verarbeitet Quaternion-Daten (optional für erweiterte Features)"""
        if len(args) >= 5:
            self.current_quaternion = {
                'w': args[1],
                'x': args[2],
                'y': args[3],
                'z': args[4]
            }
    
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
            self.server_thread.join()
    
    def predict_stroke(self):
        """Thread: Führt Vorhersagen durch wenn genug Daten vorhanden"""
        movement_threshold = 1.5  # Schwellwert für Bewegungserkennung
        cooldown_time = 1.0  # Sekunden zwischen Vorhersagen
        last_prediction_time = 0
        
        while self.running:
            if len(self.data_buffer) == self.window_size:
                current_time = time.time()
                
                # Cooldown prüfen
                if current_time - last_prediction_time > cooldown_time:
                    # Daten vorbereiten
                    window_data = np.array(list(self.data_buffer))
                    
                    # Bewegung erkennen (basierend auf Beschleunigung)
                    acc_magnitude = np.sqrt(np.sum(window_data[:, :3]**2, axis=1))
                    max_movement = np.max(acc_magnitude)
                    
                    if max_movement > movement_threshold:
                        # Normalisierung
                        try:
                            window_scaled = self.scaler.transform(window_data)
                            
                            # Vorhersage
                            X = window_scaled.reshape(1, self.window_size, 6)
                            prediction = self.model.predict(X, verbose=0)
                            predicted_class = np.argmax(prediction[0])
                            confidence = np.max(prediction[0])
                            
                            # Nur Vorhersagen mit ausreichender Konfidenz
                            if confidence > 0.3:  # 30% Mindest-Konfidenz
                                result = {
                                    'class': self.class_names[predicted_class],
                                    'confidence': confidence,
                                    'timestamp': current_time,
                                    'all_probs': prediction[0]
                                }
                                
                                self.prediction_queue.put(result)
                                last_prediction_time = current_time
                        except Exception as e:
                            print(f"Vorhersagefehler: {e}")
            
            time.sleep(0.01)  # 10ms Pause
    
    def check_connection(self, timeout=5):
        """Überprüft ob NGIMU Daten sendet"""
        print(f"\nÜberprüfe NGIMU-Verbindung für {timeout} Sekunden...")
        print(f"Lausche auf Port {self.port}...")
        
        start_time = time.time()
        initial_buffer_size = len(self.data_buffer)
        
        while time.time() - start_time < timeout:
            if len(self.data_buffer) > initial_buffer_size:
                print("✓ NGIMU sendet Daten!")
                print(f"  Empfange {len(self.data_buffer) - initial_buffer_size} Datenpunkte/Sekunde")
                return True
            time.sleep(0.1)
        
        print("✗ Keine Daten von NGIMU empfangen!")
        print("\nBitte überprüfen Sie:")
        print("1. NGIMU ist eingeschaltet")
        print("2. Sie sind mit dem NGIMU WiFi verbunden")
        print("3. NGIMU Send Settings:")
        print("   - Send IP: 192.168.1.2")
        print("   - Send Port: 8001")
        print("   - Sensors Rate: 100 Hz")
        print("   - Quaternion Rate: 50 Hz (optional)")
        return False
    
    def start_realtime_prediction(self):
        """Startet die Echtzeit-Vorhersage"""
        print("\n=== Echtzeit Tischtennisschlag-Erkennung (WiFi) ===")
        
        # Verbindung prüfen
        if not self.check_connection():
            return
        
        print("\nBereit für Vorhersagen!")
        print("Führen Sie Tischtennisschläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        
        self.running = True
        
        # Vorhersage-Thread starten
        self.predict_thread = threading.Thread(target=self.predict_stroke)
        self.predict_thread.start()
        
        # Vorhersagen anzeigen
        try:
            while True:
                if not self.prediction_queue.empty():
                    result = self.prediction_queue.get()
                    
                    print(f"\n{'='*60}")
                    print(f"⚡ SCHLAG ERKANNT!")
                    print(f"   Typ: {result['class']}")
                    print(f"   Konfidenz: {result['confidence']:.1%}")
                    
                    # Zeige alle Wahrscheinlichkeiten
                    print(f"\n   Wahrscheinlichkeiten:")
                    for i, (name, prob) in enumerate(zip(self.class_names, result['all_probs'])):
                        bar = '█' * int(prob * 20)
                        print(f"   {name:20} {bar:20} {prob:.1%}")
                    print(f"{'='*60}\n")
                
                # Status-Update alle 5 Sekunden
                if int(time.time()) % 5 == 0:
                    buffer_fill = len(self.data_buffer) / self.window_size * 100
                    print(f"\rBuffer: {buffer_fill:.0f}% | Warte auf Schlag...", end='', flush=True)
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\nBeende Vorhersage...")
            self.running = False
            if self.predict_thread:
                self.predict_thread.join()
            self.stop_server()

def main():
    """Hauptfunktion"""
    import sys
    
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