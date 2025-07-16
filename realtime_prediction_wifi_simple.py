"""
Echtzeit Tischtennisschlag-Erkennung über WiFi (Simplified Version)
Verwendet die minimalen Versionen von data_preprocessing und train_1d_cnn
Nur mit Standard-Bibliotheken: numpy, pickle, socket, threading
"""

import numpy as np
import pickle
import socket
import threading
import time
from collections import deque
import struct
import sys

# OSC Decoder ohne externe Bibliotheken
class SimpleOSCDecoder:
    """Einfacher OSC Decoder nur mit Python-Standard-Funktionen"""
    
    @staticmethod
    def decode_message(data):
        """Dekodiert OSC Nachrichten und extrahiert Sensordaten"""
        try:
            # OSC Format: Address Pattern + Type Tags + Arguments
            # Suche nach dem ersten Null-Byte (Ende der Adresse)
            addr_end = data.find(b'\x00')
            if addr_end == -1:
                return None
                
            address = data[:addr_end].decode('utf-8')
            
            # Nur /imu Nachrichten verarbeiten
            if address != '/imu':
                return None
            
            # Padding überspringen
            offset = addr_end + (4 - addr_end % 4) % 4
            
            # Type Tag String
            typetag_end = data.find(b'\x00', offset)
            typetags = data[offset:typetag_end].decode('utf-8')
            
            # Zum Beginn der Daten springen
            offset = typetag_end + (4 - typetag_end % 4) % 4
            
            # Wir erwarten 18 floats (acc, gyro, mag, quat, lin_acc, bar)
            values = []
            for i in range(18):
                if offset + 4 <= len(data):
                    # Big-Endian float entpacken
                    value = struct.unpack('>f', data[offset:offset+4])[0]
                    values.append(value)
                    offset += 4
                else:
                    values.append(0.0)
            
            # Dictionary mit Sensordaten erstellen
            sensor_data = {
                'timestamp': time.time(),
                'acc_x': values[0], 'acc_y': values[1], 'acc_z': values[2],
                'gyro_x': values[3], 'gyro_y': values[4], 'gyro_z': values[5],
                'mag_x': values[6], 'mag_y': values[7], 'mag_z': values[8],
                'quat_w': values[9], 'quat_x': values[10], 
                'quat_y': values[11], 'quat_z': values[12],
                'lin_acc_x': values[13], 'lin_acc_y': values[14], 'lin_acc_z': values[15],
                'bar': values[17] if len(values) > 17 else 0.0
            }
            
            return sensor_data
            
        except Exception as e:
            return None

class SimpleTischtennisPredictor:
    """Echtzeit-Vorhersage nur mit NumPy"""
    
    def __init__(self, model_weights_path='models/simple_model_weights.pkl',
                 normalization_path='processed_data/normalization_minimal.pkl',
                 info_path='processed_data/info_minimal.pkl'):
        
        # Modell laden
        print("Lade Modell...")
        self.load_model(model_weights_path)
        
        # Normalisierungsparameter laden
        with open(normalization_path, 'rb') as f:
            norm_params = pickle.load(f)
        self.feature_means = norm_params['means']
        self.feature_stds = norm_params['stds']
        self.features = norm_params['features']
        
        # Info laden
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
        self.stroke_types = info['stroke_types']
        self.window_size = info['window_size']
        
        # Buffer für Sensordaten
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Vorhersage-Schwellwerte
        self.confidence_threshold = 0.7
        self.min_samples_between_predictions = 50
        self.samples_since_last_prediction = 0
        
        print(f"Modell geladen!")
        print(f"Features: {self.features}")
        print(f"Schlagtypen: {self.stroke_types}")
        print(f"Window Size: {self.window_size}")
    
    def load_model(self, weights_path):
        """Lädt die gespeicherten Modell-Gewichte"""
        # Für die einfache Version: Wir verwenden ein simples Modell
        # In der Realität würden hier die trainierten Gewichte geladen
        self.model_loaded = True
        print("Vereinfachtes Modell geladen (Demo-Modus)")
    
    def simple_smooth(self, values, window=5):
        """Einfache Glättung wie im Preprocessing"""
        if len(values) < window:
            return values
        
        smoothed = np.zeros_like(values)
        half_window = window // 2
        
        smoothed[:half_window] = values[:half_window]
        smoothed[-half_window:] = values[-half_window:]
        
        for i in range(half_window, len(values) - half_window):
            smoothed[i] = np.mean(values[i-half_window:i+half_window+1])
        
        return smoothed
    
    def normalize_data(self, data):
        """Normalisiert die Daten mit gespeicherten Parametern"""
        normalized = np.zeros_like(data)
        for i in range(data.shape[1]):
            normalized[:, i] = (data[:, i] - self.feature_means[i]) / self.feature_stds[i]
        return normalized
    
    def detect_movement(self, buffer_array):
        """Erkennt ob eine Bewegung stattfindet"""
        # Bewegungsintensität berechnen
        lin_acc = np.sqrt(
            buffer_array[:, self.features.index('lin_acc_x')]**2 +
            buffer_array[:, self.features.index('lin_acc_y')]**2 +
            buffer_array[:, self.features.index('lin_acc_z')]**2
        )
        gyro = np.sqrt(
            buffer_array[:, self.features.index('gyro_x')]**2 +
            buffer_array[:, self.features.index('gyro_y')]**2 +
            buffer_array[:, self.features.index('gyro_z')]**2
        )
        
        # Intensität kombinieren
        intensity = 0.7 * lin_acc + 0.3 * gyro
        
        # Schwellwert für Bewegungserkennung
        threshold = np.mean(intensity) + 0.5 * np.std(intensity)
        max_intensity = np.max(intensity[-20:])  # Letzte 20 Samples
        
        return max_intensity > threshold
    
    def simple_predict(self, data):
        """Sehr einfache Vorhersage basierend auf Bewegungsmustern"""
        # Dies ist eine Demo-Implementierung
        # In der echten Version würde hier das trainierte CNN verwendet
        
        # Feature-Extraktion
        gyro_x_mean = np.mean(data[:, self.features.index('gyro_x')])
        gyro_z_mean = np.mean(data[:, self.features.index('gyro_z')])
        lin_acc_y_max = np.max(data[:, self.features.index('lin_acc_y')])
        
        # Einfache Regel-basierte Klassifikation als Platzhalter
        if gyro_x_mean > 0.5 and lin_acc_y_max > 5:
            prediction = 0  # Vorhand Topspin
            confidence = 0.8
        elif gyro_x_mean < -0.5 and lin_acc_y_max > 5:
            prediction = 2  # Rückhand Topspin
            confidence = 0.8
        elif gyro_z_mean > 0.3:
            prediction = 1  # Vorhand Schupf
            confidence = 0.75
        else:
            prediction = 3  # Rückhand Schupf
            confidence = 0.75
        
        # Pseudo-Wahrscheinlichkeiten
        probs = np.zeros(4)
        probs[prediction] = confidence
        remaining = 1 - confidence
        for i in range(4):
            if i != prediction:
                probs[i] = remaining / 3
        
        return probs
    
    def process_sample(self, sensor_data):
        """Verarbeitet eine neue Sensorprobe"""
        # Features extrahieren
        sample = []
        for feature in self.features:
            sample.append(sensor_data.get(feature, 0.0))
        
        # Zum Buffer hinzufügen
        self.data_buffer.append(sample)
        
        # Vorhersage nur wenn Buffer voll ist
        if len(self.data_buffer) < self.window_size:
            return None
        
        # Mindestabstand zwischen Vorhersagen
        self.samples_since_last_prediction += 1
        if self.samples_since_last_prediction < self.min_samples_between_predictions:
            return None
        
        # Buffer in Array konvertieren
        buffer_array = np.array(self.data_buffer)
        
        # Bewegungserkennung
        if not self.detect_movement(buffer_array):
            return None
        
        # Daten glätten
        for i, feature in enumerate(self.features):
            if 'gyro' in feature or 'lin_acc' in feature:
                buffer_array[:, i] = self.simple_smooth(buffer_array[:, i])
        
        # Normalisieren
        normalized_data = self.normalize_data(buffer_array)
        
        # Vorhersage
        predictions = self.simple_predict(normalized_data)
        
        # Beste Klasse finden
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Nur ausgeben wenn Konfidenz hoch genug
        if confidence >= self.confidence_threshold:
            self.samples_since_last_prediction = 0
            return {
                'stroke_type': self.stroke_types[predicted_class],
                'confidence': confidence,
                'all_probabilities': predictions
            }
        
        return None

class WiFiSensorReceiver:
    """Empfängt Sensordaten über WiFi"""
    
    def __init__(self, host='0.0.0.0', port=9000):
        self.host = host
        self.port = port
        self.sock = None
        self.running = False
        self.predictor = SimpleTischtennisPredictor()
        self.last_prediction_time = 0
        self.min_time_between_predictions = 1.0  # Sekunden
        
    def start(self):
        """Startet den WiFi-Empfänger"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.1)  # Non-blocking mit timeout
            
            print(f"\nWiFi-Empfänger gestartet auf {self.host}:{self.port}")
            print("Warte auf Sensordaten...")
            print("Drücke Strg+C zum Beenden\n")
            
            self.running = True
            self.receive_loop()
            
        except Exception as e:
            print(f"Fehler beim Starten: {e}")
        finally:
            if self.sock:
                self.sock.close()
    
    def receive_loop(self):
        """Hauptschleife zum Empfangen und Verarbeiten"""
        sample_count = 0
        
        while self.running:
            try:
                # Daten empfangen
                data, addr = self.sock.recvfrom(1024)
                
                # OSC dekodieren
                sensor_data = SimpleOSCDecoder.decode_message(data)
                
                if sensor_data:
                    sample_count += 1
                    
                    # Vorhersage versuchen
                    result = self.predictor.process_sample(sensor_data)
                    
                    if result:
                        current_time = time.time()
                        if current_time - self.last_prediction_time >= self.min_time_between_predictions:
                            self.display_prediction(result)
                            self.last_prediction_time = current_time
                    
                    # Status alle 1000 Samples
                    if sample_count % 1000 == 0:
                        print(f"[Status] {sample_count} Samples empfangen...")
                
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                print("\nBeende Programm...")
                self.running = False
            except Exception as e:
                print(f"Fehler: {e}")
    
    def display_prediction(self, result):
        """Zeigt die Vorhersage an"""
        print("\n" + "="*50)
        print(f"🏓 SCHLAG ERKANNT: {result['stroke_type']}")
        print(f"   Konfidenz: {result['confidence']:.2%}")
        print(f"   Zeit: {time.strftime('%H:%M:%S')}")
        
        # Wahrscheinlichkeiten für alle Klassen
        print("\n   Alle Wahrscheinlichkeiten:")
        for i, (stroke, prob) in enumerate(zip(self.predictor.stroke_types, 
                                               result['all_probabilities'])):
            bar = "█" * int(prob * 20)
            print(f"   {stroke:20} {prob:5.1%} {bar}")
        
        print("="*50 + "\n")

def main():
    """Hauptfunktion"""
    print("=== TISCHTENNIS ECHTZEIT-ERKENNUNG (Simplified) ===")
    print("Verwendet vereinfachte Verarbeitung ohne externe ML-Bibliotheken")
    print("Dies ist eine Demo-Version mit regel-basierter Klassifikation")
    print("")
    
    # Parameter anzeigen
    print("Konfiguration:")
    print(f"- Host: 0.0.0.0")
    print(f"- Port: 9000")
    print(f"- Window Size: 200 Samples")
    print(f"- Konfidenz-Schwelle: 70%")
    print("")
    
    # Empfänger starten
    receiver = WiFiSensorReceiver()
    
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("\nProgramm beendet.")
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    main()
