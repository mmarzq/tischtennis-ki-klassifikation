"""
Echtzeit-Vorhersage für Tischtennisschläge
Verwendet das trainierte Modell zur Live-Klassifikation
"""

import numpy as np
import tensorflow as tf
import serial
import pickle
import time
from collections import deque
import threading
import queue

class RealtimePredictor:
    def __init__(self, model_path, scaler_path, port='COM4', baudrate=115200):
        # Modell laden
        self.model = tf.keras.models.load_model(model_path)
        
        # Scaler laden
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Modell-Info laden
        with open('models/model_info.pkl', 'rb') as f:
            self.model_info = pickle.load(f)
        
        self.class_names = self.model_info['class_names']
        
        # Serial Verbindung
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        
        # Daten-Buffer
        self.window_size = 200  # 2 Sekunden @ 100Hz
        self.data_buffer = deque(maxlen=self.window_size)
        self.prediction_queue = queue.Queue()
        
        # Threading
        self.running = False
        self.read_thread = None
        self.predict_thread = None
        
    def connect(self):
        """Verbindung zum Arduino herstellen"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)
            print(f"Verbunden mit {self.port}")
            return True
        except Exception as e:
            print(f"Verbindungsfehler: {e}")
            return False
    
    def disconnect(self):
        """Verbindung trennen"""
        self.running = False
        if self.read_thread:
            self.read_thread.join()
        if self.predict_thread:
            self.predict_thread.join()
        if self.ser:
            self.ser.close()
    
    def read_sensor_data(self):
        """Thread: Liest kontinuierlich Sensordaten"""
        while self.running:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line and ',' in line:
                        values = line.split(',')
                        if len(values) >= 7:
                            # Timestamp ignorieren, nur Sensorwerte
                            sensor_values = [float(v) for v in values[1:7]]
                            self.data_buffer.append(sensor_values)
                except Exception as e:
                    print(f"Lesefehler: {e}")
    
    def predict_stroke(self):
        """Thread: Führt Vorhersagen durch wenn genug Daten vorhanden"""
        movement_threshold = 2.0  # Schwellwert für Bewegungserkennung
        cooldown_time = 1.0  # Sekunden zwischen Vorhersagen
        last_prediction_time = 0
        
        while self.running:
            if len(self.data_buffer) == self.window_size:
                current_time = time.time()
                
                # Cooldown prüfen
                if current_time - last_prediction_time > cooldown_time:
                    # Daten vorbereiten
                    window_data = np.array(list(self.data_buffer))
                    
                    # Bewegung erkennen
                    acc_magnitude = np.sqrt(np.sum(window_data[:, :3]**2, axis=1))
                    max_movement = np.max(acc_magnitude)
                    
                    if max_movement > movement_threshold:
                        # Normalisierung
                        window_scaled = self.scaler.transform(window_data)
                        
                        # Vorhersage
                        X = window_scaled.reshape(1, self.window_size, 6)
                        prediction = self.model.predict(X, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class]
                        
                        # Ergebnis
                        result = {
                            'class': self.class_names[predicted_class],
                            'confidence': confidence,
                            'timestamp': current_time
                        }
                        
                        self.prediction_queue.put(result)
                        last_prediction_time = current_time
            
            time.sleep(0.01)  # 10ms Pause
    
    def start_realtime_prediction(self):
        """Startet die Echtzeit-Vorhersage"""
        if not self.ser:
            print("Keine Verbindung!")
            return
        
        print("\n=== Echtzeit Tischtennisschlag-Erkennung ===")
        print("Führen Sie Schläge aus...")
        print("Drücken Sie Ctrl+C zum Beenden\n")
        
        self.running = True
        
        # Threads starten
        self.read_thread = threading.Thread(target=self.read_sensor_data)
        self.predict_thread = threading.Thread(target=self.predict_stroke)
        
        self.read_thread.start()
        self.predict_thread.start()
        
        # Vorhersagen anzeigen
        try:
            while True:
                if not self.prediction_queue.empty():
                    result = self.prediction_queue.get()
                    print(f"\n{'='*50}")
                    print(f"ERKANNT: {result['class']}")
                    print(f"Konfidenz: {result['confidence']:.2%}")
                    print(f"{'='*50}\n")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nBeende Vorhersage...")
            self.disconnect()

def demo_mode():
    """Demo-Modus mit simulierten Daten"""
    print("\n=== DEMO MODUS ===")
    print("Simuliere Tischtennisschläge...\n")
    
    # Modell laden
    model = tf.keras.models.load_model('models/best_model.h5')
    
    # Simulierte Schlagmuster
    patterns = {
        'Vorhand Topspin': lambda t: [
            0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.1),
            0.3 * np.cos(2 * np.pi * 5 * t) + np.random.normal(0, 0.1),
            1.0 + 0.2 * np.sin(2 * np.pi * 2 * t),
            10 * np.sin(2 * np.pi * 8 * t),
            5 * np.cos(2 * np.pi * 8 * t),
            2 * np.sin(2 * np.pi * 3 * t)
        ],
        'Rückhand Schupf': lambda t: [
            0.3 * np.sin(2 * np.pi * 3 * t) + np.random.normal(0, 0.05),
            0.2 * np.cos(2 * np.pi * 3 * t) + np.random.normal(0, 0.05),
            1.0 + 0.1 * np.sin(2 * np.pi * 1 * t),
            5 * np.sin(2 * np.pi * 4 * t),
            3 * np.cos(2 * np.pi * 4 * t),
            1 * np.sin(2 * np.pi * 2 * t)
        ]
    }
    
    # Scaler laden
    with open('processed_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    for stroke_name, pattern_func in patterns.items():
        print(f"\nSimuliere: {stroke_name}")
        
        # Daten generieren
        t = np.linspace(0, 2, 200)  # 2 Sekunden
        data = np.array([pattern_func(ti) for ti in t])
        
        # Normalisieren und vorhersagen
        data_scaled = scaler.transform(data)
        X = data_scaled.reshape(1, 200, 6)
        
        prediction = model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        print(f"Vorhersage: {['Vorhand Topspin', 'Vorhand Schupf', 'Rückhand Topspin', 'Rückhand Schupf'][predicted_class]}")
        print(f"Konfidenz: {prediction[0][predicted_class]:.2%}")
        
        time.sleep(2)

def main():
    """Hauptfunktion"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        # Echtzeit-Modus
        predictor = RealtimePredictor(
            model_path='models/best_model.h5',
            scaler_path='processed_data/scaler.pkl',
            port='COM4',
            baudrate=115200
        )
        
        if predictor.connect():
            predictor.start_realtime_prediction()
        else:
            print("Konnte keine Verbindung herstellen!")
            print("Starten Sie den Demo-Modus mit: python realtime_prediction.py --demo")

if __name__ == "__main__":
    main()
