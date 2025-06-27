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
        
        # Spaltenamen normalisieren - beide Formate unterstützen
        col_mapping = {
            'Timestamp': 'timestamp',
            'Acc_X': 'acc_x',
            'Acc_Y': 'acc_y', 
            'Acc_Z': 'acc_z',
            'Gyro_X': 'gyro_x',
            'Gyro_Y': 'gyro_y',
            'Gyro_Z': 'gyro_z',
            'Quat_W': 'quat_w',
            'Quat_X': 'quat_x',
            'Quat_Y': 'quat_y',
            'Quat_Z': 'quat_z'
        }
        
        # Spalten umbenennen falls nötig
        df.rename(columns=col_mapping, inplace=True)
        
        # Prüfen ob bereits normalisierte Spalten vorhanden sind
        expected_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        if not all(col in df.columns for col in expected_cols[:7]):
            # Versuche Spalten umzubenennen (für Rohdaten ohne Header)
            if len(df.columns) >= 7:
                df.columns = expected_cols[:len(df.columns)]
            else:
                print(f"Warnung: Unerwartete Spaltenanzahl in {filepath}")
                print(f"Gefundene Spalten: {list(df.columns)}")
        
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
            if col in filtered_data.columns:
                filtered_data[col] = filtfilt(b, a, data[col])
        
        return filtered_data
    
    def detect_strokes(self, data, threshold_factor=0.5):
        """Erkennt einzelne Schläge basierend auf Bewegungsintensität"""
        """
        Für Anfänger vielleicht threshold_factor=1.2 (erkennt auch schwächere Schläge),
        default war 1.5 
        für Profis threshold_factor=2.0 (nur klare, starke Schläge).
        """

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

        """ Algorithmus v0 """   
        """     
        # Kombinierte Bewegungsintensität
        movement_intensity = acc_magnitude + (gyro_magnitude / 100)
        #movement_intensity = acc_magnitude + (gyro_magnitude / 100) # 100: Skalierung (empirische Wahl)
        
        Der Faktor 100 ist eine empirische Wahl:
            Bringt beide Signale in vergleichbare Größenordnungen
            Verhindert, dass die Winkelgeschwindigkeit die Gesamtintensität dominiert
            Ermöglicht, dass beide Sensortypen zur Schlagerkennung beitragen
        """

        """ Algorithmus v1 """
        
        # Normalisierung (Z-Score)
        acc_norm = (acc_magnitude - acc_magnitude.mean()) / acc_magnitude.std()
        gyro_norm = (gyro_magnitude - gyro_magnitude.mean()) / gyro_magnitude.std()
        
        # Gewichtete Kombination
        w_acc = 0.9
        w_gyro = 0.1
        movement_intensity = w_acc * acc_norm + w_gyro * gyro_norm
        

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
            try:
                # Daten laden und filtern
                data = self.load_raw_data(file)
                
                if len(data) == 0:
                    print(f"  {os.path.basename(file)}: Keine Daten gefunden")
                    continue
                    
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
                    
            except Exception as e:
                print(f"  Fehler bei {os.path.basename(file)}: {str(e)}")
                continue

        """
        all_windows = [
            DataFrame1,  # 1. erkannter Schlag (200 Zeilen × 7 Spalten)
            DataFrame2,  # 2. erkannter Schlag (200 Zeilen × 7 Spalten)
            DataFrame3,  # 3. erkannter Schlag (200 Zeilen × 7 Spalten)
            # ... weitere Schläge
        ]

        # windows[0] könnte so aussehen:
            timestamp    acc_x    acc_y    acc_z   gyro_x   gyro_y   gyro_z
        0     1.23       0.1      0.2      9.8      5.2      1.1      0.3
        1     1.24       0.3      0.4      9.7      8.1      2.2      1.1
        2     1.25       1.2      2.1      8.9     15.3      5.4      3.2
        ...   ...        ...      ...      ...      ...      ...      ...
        199   3.22       0.2      0.1      9.8      2.1      0.8      0.2
        """
        
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
    """
    label_map = {}
    for idx, stroke in enumerate(stroke_types):
        label_map[stroke] = idx
    
    # label_map sieht so aus:
    label_map = {
        'vorhand_topspin': 0,
        'vorhand_schupf': 1,
        'rueckhand_topspin': 2,
        'rueckhand_schupf': 3
    }
    """

    all_data = []
    all_labels = []
    
    # Verarbeite jeden Schlagtyp
    for stroke_type in stroke_types:
        windows, features = processor.process_stroke_type(stroke_type, './rohdaten')
        
        # Labels hinzufügen
        labels = [label_map[stroke_type]] * len(windows)
        """ 
        labels = [0] * 5    # Beispiel: 5 Schläge => len(windows) = 5 
        # Ergebnis: labels = [0, 0, 0, 0, 0]
        """
        
        all_data.extend(windows)
        all_labels.extend(labels)
    
    # In numpy arrays konvertieren
    if len(all_data) > 0:
        X = np.array(all_data)
        y = np.array(all_labels)
        
        print(f"\nGesamtdaten: {X.shape[0]} Schläge")
        print(f"Form: {X.shape}")
        print(f"Klassen: {np.unique(y, return_counts=True)}")
        
        # Ordner erstellen falls nicht vorhanden
        os.makedirs('./processed_data', exist_ok=True)
        
        # Speichern
        np.save('./processed_data/X_data.npy', X)
        np.save('./processed_data/y_labels.npy', y)
        
        # Scaler fit und speichern
        X_reshaped = X.reshape(-1, X.shape[-1])     # (*)
        processor.scaler.fit(X_reshaped)

        """ (*)
        # Vorher: X.shape = (100, 200, 6) für 100 Schläge, je 200 Zeitpunkte (Timestamp), je 6 Sensordaten 
        # 3D-Array mit 100×200×6 = 120.000 Datenpunkten
            Was macht X.reshape(-1, X.shape[-1])?
            1. X.shape[-1] = letzter Wert der Shape = 6 (Anzahl Sensoren)
            2. X.reshape(-1, 6) formt das 3D-Array in ein 2D-Array um
                Bsp 1: X.reshape(-1, 6):
                    # -1 wird automatisch berechnet: 120.000 ÷ 6 = 20.000
                    # Ergebnis: (20000, 6)
                Bsp 2: X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
                    # X.shape = (12,)
                    X_reshaped = X.reshape(2, 6)
                    # Ergebnis: [[1,  2,  3,  4,  5,  6], [7,  8,  9, 10, 11, 12]]
                    # X_reshaped.shape = (2, 6)
            3. -1 bedeutet: "Berechne diese Dimension automatisch"
        # Nachher: X_reshaped.shape = (20000, 6)
        # 2D-Array mit 20.000 Zeilen × 6 Spalten
        # (100 Schläge × 200 Zeitpunkte = 20.000 Zeilen)
            Warum wird das gemacht?
            Der StandardScaler normalisiert spaltenweise. Er kann nur mit 2D-Arrays arbeiten, wo:
            Jede Zeile = ein Beobachtung/Messung
            Jede Spalte = ein Feature/Sensor (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        """
        
        with open('./processed_data/scaler.pkl', 'wb') as f:
            pickle.dump(processor.scaler, f)
        
        print("\nDaten gespeichert in processed_data/")
        
        return X, y
    else:
        print("\nKeine Daten zum Verarbeiten gefunden!")
        return np.array([]), np.array([])

if __name__ == "__main__":
    # Beispiel: Einzelne Datei visualisieren
    processor = TischtennisDataProcessor()
    
    # Teste mit einer Beispieldatei
    test_files = glob.glob("./rohdaten/vorhand_topspin/*.csv")
    if test_files:
        test_file = test_files[0]
        print(f"Teste mit: {test_file}")
        try:
            data = processor.load_raw_data(test_file)
            filtered = processor.apply_filters(data)
            peaks, intensity = processor.detect_strokes(filtered)
            
            # Ordner für Visualisierungen erstellen
            os.makedirs('./visualizations', exist_ok=True)
            
            processor.visualize_stroke_detection(filtered, peaks, intensity, 
                                               "./visualizations/stroke_detection_example.png")
        except Exception as e:
            print(f"Fehler bei Visualisierung: {e}")
    
    # Alle Daten verarbeiten
    X, y = process_all_data()
