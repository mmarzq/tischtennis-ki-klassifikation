"""
Datenvorverarbeitung für Tischtennisschlag-Klassifikation
Segmentiert Rohdaten und bereitet sie für 1D CNN vor
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy import signal
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

class TischtennisDataProcessor:
    def __init__(self, window_size=200, overlap=50):
        self.window_size = window_size  # 200 samples @ 100Hz = 2 Sekunden
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def load_raw_data(self, filepath):
        """Lädt CSV-Rohdaten (kompatibel mit Arduino und NGIMU)"""
        df = pd.read_csv(filepath)
        
        # Spaltenamen normalisieren
        # Prüfen ob bereits normalisierte Spalten vorhanden sind
        expected_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        if all(col in df.columns for col in expected_cols[:7]):
            # Bereits im richtigen Format
            pass
        else:
            # Versuche Spalten umzubenennen (für Rohdaten)
            if len(df.columns) >= 7:
                df.columns = expected_cols[:len(df.columns)]
            else:
                print(f"Warnung: Unerwartete Spaltenanzahl in {filepath}")
        
        # Nur relevante Spalten behalten
        sensor_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        available_cols = [col for col in sensor_cols if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def apply_filters(self, data, cutoff_freq=20, fs=100):
        """Wendet Butterworth-Tiefpassfilter an"""
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
        filtered_data = data.copy()
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        for col in sensor_cols:
            filtered_data[col] = filtfilt(b, a, data[col])
        
        return filtered_data
    
    def detect_strokes(self, data, threshold_factor=1.5):
        """Erkennt einzelne Schläge basierend auf Bewegungsintensität"""
        # Berechne Gesamtbeschleunigung
        acc_magnitude = np.sqrt(
            data['acc_x']**2 + 
            data['acc_y']**2 + 
            data['acc_z']**2
        )
        
        # Berechne Winkelgeschwindigkeit
        gyro_magnitude = np.sqrt(
            data['gyro_x']**2 + 
            data['gyro_y']**2 + 
            data['gyro_z']**2
        )
        
        # Kombinierte Bewegungsintensität
        movement_intensity = acc_magnitude + (gyro_magnitude / 100)
        
        # Schwellwert basierend auf Standardabweichung
        threshold = movement_intensity.mean() + threshold_factor * movement_intensity.std()
        
        # Finde Peaks
        peaks, properties = find_peaks(
            movement_intensity, 
            height=threshold,
            distance=50,  # Mindestabstand zwischen Schlägen (0.5s @ 100Hz)
            prominence=threshold/2
        )
        
        return peaks, movement_intensity
    
    def extract_stroke_windows(self, data, peaks):
        """Extrahiert Fenster um erkannte Schläge"""
        windows = []
        
        for peak in peaks:
            # Fenster zentriert um Peak
            start = max(0, peak - self.window_size // 2)
            end = min(len(data), peak + self.window_size // 2)
            
            # Nur vollständige Fenster verwenden
            if end - start == self.window_size:
                window = data.iloc[start:end].copy()
                window.reset_index(drop=True, inplace=True)
                windows.append(window)
        
        return windows
    
    def calculate_features(self, window):
        """Berechnet zusätzliche Features für ein Fenster"""
        features = {}
        
        # Basis-Features bleiben die Rohdaten
        sensor_data = window[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
        
        # Zusätzliche abgeleitete Features (optional)
        features['max_acc'] = np.sqrt(
            window['acc_x']**2 + 
            window['acc_y']**2 + 
            window['acc_z']**2
        ).max()
        
        features['max_gyro'] = np.sqrt(
            window['gyro_x']**2 + 
            window['gyro_y']**2 + 
            window['gyro_z']**2
        ).max()
        
        return sensor_data, features
    
    def process_stroke_type(self, stroke_type, input_folder):
        """Verarbeitet alle Dateien eines Schlagtyps"""
        all_windows = []
        all_features = []
        
        # Alle CSV-Dateien für diesen Schlagtyp
        files = glob.glob(f"{input_folder}/{stroke_type}/*.csv")
        
        print(f"\nVerarbeite {stroke_type}: {len(files)} Dateien gefunden")
        
        for file in files:
            # Daten laden und filtern
            data = self.load_raw_data(file)
            filtered_data = self.apply_filters(data)
            
            # Schläge erkennen
            peaks, intensity = self.detect_strokes(filtered_data)
            
            # Fenster extrahieren
            windows = self.extract_stroke_windows(filtered_data, peaks)
            
            print(f"  {os.path.basename(file)}: {len(windows)} Schläge erkannt")
            
            # Features berechnen
            for window in windows:
                sensor_data, features = self.calculate_features(window)
                all_windows.append(sensor_data)
                all_features.append(features)
        
        return all_windows, all_features
    
    def visualize_stroke_detection(self, data, peaks, intensity, output_file=None):
        """Visualisiert die Schlagerkennung"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Beschleunigung
        axes[0].plot(data['acc_x'], label='X', alpha=0.7)
        axes[0].plot(data['acc_y'], label='Y', alpha=0.7)
        axes[0].plot(data['acc_z'], label='Z', alpha=0.7)
        axes[0].set_ylabel('Beschleunigung (g)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Winkelgeschwindigkeit
        axes[1].plot(data['gyro_x'], label='X', alpha=0.7)
        axes[1].plot(data['gyro_y'], label='Y', alpha=0.7)
        axes[1].plot(data['gyro_z'], label='Z', alpha=0.7)
        axes[1].set_ylabel('Winkelgeschw. (°/s)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Bewegungsintensität mit erkannten Peaks
        axes[2].plot(intensity, label='Bewegungsintensität')
        axes[2].plot(peaks, intensity[peaks], 'ro', markersize=8, label='Erkannte Schläge')
        axes[2].set_ylabel('Intensität')
        axes[2].set_xlabel('Samples')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        plt.show()

def process_all_data():
    """Hauptfunktion zur Verarbeitung aller Daten"""
    processor = TischtennisDataProcessor(window_size=200, overlap=50)
    
    stroke_types = ['vorhand_topspin', 'vorhand_schupf', 'rueckhand_topspin', 'rueckhand_schupf']
    label_map = {stroke: idx for idx, stroke in enumerate(stroke_types)}
    
    all_data = []
    all_labels = []
    
    # Verarbeite jeden Schlagtyp
    for stroke_type in stroke_types:
        windows, features = processor.process_stroke_type(stroke_type, '../rohdaten')
        
        # Labels hinzufügen
        labels = [label_map[stroke_type]] * len(windows)
        
        all_data.extend(windows)
        all_labels.extend(labels)
    
    # In numpy arrays konvertieren
    X = np.array(all_data)
    y = np.array(all_labels)
    
    print(f"\nGesamtdaten: {X.shape[0]} Schläge")
    print(f"Form: {X.shape}")
    print(f"Klassen: {np.unique(y, return_counts=True)}")
    
    # Speichern
    np.save('../processed_data/X_data.npy', X)
    np.save('../processed_data/y_labels.npy', y)
    
    # Scaler fit und speichern
    X_reshaped = X.reshape(-1, X.shape[-1])
    processor.scaler.fit(X_reshaped)
    
    with open('../processed_data/scaler.pkl', 'wb') as f:
        pickle.dump(processor.scaler, f)
    
    print("\nDaten gespeichert in processed_data/")
    
    return X, y

if __name__ == "__main__":
    # Beispiel: Einzelne Datei visualisieren
    processor = TischtennisDataProcessor()
    
    # Teste mit einer Beispieldatei (passen Sie den Pfad an)
    test_file = "../rohdaten/vorhand_topspin/example.csv"
    if os.path.exists(test_file):
        data = processor.load_raw_data(test_file)
        filtered = processor.apply_filters(data)
        peaks, intensity = processor.detect_strokes(filtered)
        processor.visualize_stroke_detection(filtered, peaks, intensity, 
                                           "../visualizations/stroke_detection_example.png")
    
    # Alle Daten verarbeiten
    X, y = process_all_data()
