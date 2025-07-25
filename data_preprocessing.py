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
        
        # Features die wir für das Training verwenden
        self.training_features = [
            'gyro_x', 'gyro_y', 'gyro_z',
            'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
            'quat_w', 'quat_x', 'quat_y', 'quat_z'
        ]
        
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
            'Mag_X': 'mag_x',
            'Mag_Y': 'mag_y',
            'Mag_Z': 'mag_z',
            'Bar': 'bar',
            'Quat_W': 'quat_w',
            'Quat_X': 'quat_x',
            'Quat_Y': 'quat_y',
            'Quat_Z': 'quat_z',
            'Lin_Acc_X': 'lin_acc_x',
            'Lin_Acc_Y': 'lin_acc_y',
            'Lin_Acc_Z': 'lin_acc_z'
        }
        
        # Spalten umbenennen falls nötig
        df.rename(columns=col_mapping, inplace=True)
        
        # Prüfen ob alle benötigten Spalten vorhanden sind
        required_cols = ['timestamp'] + self.training_features
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warnung: Fehlende Spalten in {filepath}: {missing_cols}")
            # Für fehlende Spalten Nullen einfügen
            for col in missing_cols:
                df[col] = 0.0
        
        # Nur relevante Spalten behalten
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def apply_filters(self, data, cutoff_freq=20, fs=100):
        """Wendet Butterworth-Tiefpassfilter an"""
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        
        filtered_data = data.copy()
        
        # Filter nur auf kontinuierliche Sensordaten anwenden (nicht auf Quaternions)
        filter_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']
        
        for col in filter_cols:
            if col in filtered_data.columns:
                filtered_data[col] = filtfilt(b, a, data[col])
        
        return filtered_data
    
    def detect_strokes(self, data, threshold_factor=0.5):
        """Erkennt einzelne Schläge basierend auf Bewegungsintensität"""
        # Berechne Bewegungsintensität aus linearer Beschleunigung und Gyroskop
        lin_acc_magnitude = np.sqrt(
            data['lin_acc_x']**2 + 
            data['lin_acc_y']**2 + 
            data['lin_acc_z']**2
        )
        
        gyro_magnitude = np.sqrt(
            data['gyro_x']**2 + 
            data['gyro_y']**2 + 
            data['gyro_z']**2
        )
        
        """ Algorithmus v0: einfache Kombination der Magnituden"""   
        """     
        # Kombinierte Bewegungsintensität
        #movement_intensity = lin_acc_magnitude + (gyro_magnitude / 100) # 100: Skalierung (empirische Wahl)
        
        Der Faktor 100 ist eine empirische Wahl:
            Bringt beide Signale in vergleichbare Größenordnungen
            Verhindert, dass die Winkelgeschwindigkeit die Gesamtintensität dominiert
            Ermöglicht, dass beide Sensortypen zur Schlagerkennung beitragen
        """

        """ Algorithmus v1: gewichtete Kombination der normalisierten (Z-Wert) Magnituden"""
        """
        A z-score, also known as a standard score, is a statistical measurement that describes 
        the position of a raw score in terms of its distance from the mean, measured in standard deviations. 
        It indicates how many standard deviations a data point is away from the mean of a distribution. 
            - Positive z-scores indicate values above the mean, 
            - while negative z-scores indicate values below the mean. 
        Formula:
            z = (x - μ) / σ, where 'x' is the raw score, 'μ' is the population mean, and 'σ' is the population standard deviation. 
        Purpose:
            Z-scores help standardize data, allowing for comparisons between different distributions and the identification of outliers. 
        Interpretation:
            A z-score of 0 means the data point is equal to the mean. A z-score of +1 means the data point is one standard deviation above the mean. A z-score of -2 means the data point is two standard deviations below the mean. 
        Applications:
            Z-scores are used in various statistical analyses, including hypothesis testing, probability estimation, and identifying unusual data points. 
        Outliers:
            Data points with very high or very low z-scores (e.g., beyond ±2 or ±3) are often considered outliers. 
        Source: Google Gemini
        """
        # Normalisierung (Z-Wert-Normalisierung)
        lin_acc_norm = (lin_acc_magnitude - lin_acc_magnitude.mean()) / (lin_acc_magnitude.std() + 1e-6)
        gyro_norm = (gyro_magnitude - gyro_magnitude.mean()) / (gyro_magnitude.std() + 1e-6)
        
        # Gewichtete Kombination
        w_acc = 0.8  # Lineare Beschleunigung ist wichtiger
        w_gyro = 0.2
        movement_intensity = w_acc * lin_acc_norm + w_gyro * gyro_norm # obere Berechnungen sind zur Ermittlung der Bewegungsintensität 
        
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
        """ Extrahiert Fenster (geschnittetes Data) um erkannte Schläge"""
        windows = []
        
        for peak in peaks:
            # Fenster zentriert um Peak
            start = max(0, peak - self.window_size // 2)
            end = min(len(data), peak + self.window_size // 2)
            
            # Nur vollständige Fenster verwenden
            if end - start == self.window_size: # das heißt: en Schlag muss ungefähr in der Mitte sein (zwischen 2. bis 4.Minute)
                window = data.iloc[start:end].copy()
                window.reset_index(drop=True, inplace=True)
                windows.append(window)
        
        return windows
    
    def calculate_features(self, window):
        """Berechnet Features für ein Fenster"""
        # Extrahiere nur die Features die wir für das Training verwenden
        sensor_data = window[self.training_features].values #Timestamps gehöhren nicht dazu 
        
        # Zusätzliche abgeleitete Features können hier berechnet werden
        features = {}
        
        # Bewegungsintensität aus linearer Beschleunigung
        features['max_lin_acc'] = np.sqrt(
            window['lin_acc_x']**2 + 
            window['lin_acc_y']**2 + 
            window['lin_acc_z']**2
        ).max()
        
        # Maximale Winkelgeschwindigkeit
        features['max_gyro'] = np.sqrt(
            window['gyro_x']**2 + 
            window['gyro_y']**2 + 
            window['gyro_z']**2
        ).max()
        
        return sensor_data, features
    
    def process_stroke_type(self, stroke_type, input_folder='./rohdaten', output_folder='./processed_data'):
        """Verarbeitet alle Dateien eines Schlagtyps"""
        all_windows = []
        all_features = []
        
        # Alle CSV-Dateien für diesen Schlagtyp
        files = glob.glob(f"{input_folder}/{stroke_type}/*.csv")
        
        #Ordner für Visualisierungen erstellen
        #visual_processed_data_folder = f'{output_folder}/visual_processed/{stroke_type}'
        visual_processed_data_folder = os.path.join(output_folder, 'visual_processed', stroke_type)
        os.makedirs(visual_processed_data_folder, exist_ok=True)
        
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
                
                #visualisiere
                filename = os.path.basename(file)
                visualization_path = os.path.join(visual_processed_data_folder, f"{filename}.png")
                self.visualize_stroke_detection(filtered_data, peaks, intensity, visualization_path)
                
                # Features berechnen
                #for window in windows:
                for i, window in enumerate(windows):
                    #Visualisiere window (geschnittene data inkl. Timestamps, Mitte ist ein peak)
                    window_name = os.path.join(visual_processed_data_folder, f"{filename}_window_{i}.png")
                    self.visualize_stroke_detection(window, output_file=window_name)
                    
                    sensor_data, features = self.calculate_features(window) #Merkmale, sensor_data: window (ohne Timestamps), weitere features: 'max_lin_acc', 'max_gyro'
                    all_windows.append(sensor_data)
                    all_features.append(features)
                    
            except Exception as e:
                print(f"  Fehler bei {os.path.basename(file)}: {str(e)}")
                continue
        
        #all_windows: alle window (geschnittete data (exl.Timestamps), Mitte ist ein peak)
        #all_features: alle weitere MErkmale(features): 'max_lin_acc', 'max_gyro'

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
    
    def visualize_stroke_detection(self, data, peaks=None, intensity=None, output_file=None):
        """Visualisiert die Schlagerkennung"""
        # Prüfe ob peaks und intensity vorhanden sind
        show_stroke_detection = peaks is not None and intensity is not None
        num_plots = 4 if show_stroke_detection else 3
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 12))
        
        # Gyroskop
        axes[0].plot(data['gyro_x'], label='X', alpha=0.7)
        axes[0].plot(data['gyro_y'], label='Y', alpha=0.7)
        axes[0].plot(data['gyro_z'], label='Z', alpha=0.7)
        axes[0].set_ylabel('Winkelgeschw. (°/s)')
        axes[0].set_title('Gyroskop')
        axes[0].legend()
        axes[0].grid(True)
        
        # Lineare Beschleunigung
        axes[1].plot(data['lin_acc_x'], label='X', alpha=0.7)
        axes[1].plot(data['lin_acc_y'], label='Y', alpha=0.7)
        axes[1].plot(data['lin_acc_z'], label='Z', alpha=0.7)
        axes[1].set_ylabel('Lin. Beschl. (g)')
        axes[1].set_title('Lineare Beschleunigung')
        axes[1].legend()
        axes[1].grid(True)
        
        # Quaternion
        axes[2].plot(data['quat_w'], label='W', alpha=0.7)
        axes[2].plot(data['quat_x'], label='X', alpha=0.7)
        axes[2].plot(data['quat_y'], label='Y', alpha=0.7)
        axes[2].plot(data['quat_z'], label='Z', alpha=0.7)
        axes[2].set_ylabel('Quaternion')
        axes[2].set_title('Orientierung (Quaternion)')
        axes[2].legend()
        axes[2].grid(True)
        
        # Bewegungsintensität mit erkannten Peaks
        if show_stroke_detection:
            axes[3].plot(intensity, label='Bewegungsintensität')
            axes[3].plot(peaks, intensity[peaks], 'ro', markersize=8, label='Erkannte Schläge')
            axes[3].set_ylabel('Intensität')
            axes[3].set_xlabel('Samples')
            axes[3].set_title('Schlagerkennung')
            axes[3].legend()
            axes[3].grid(True)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            plt.close()  #Schließe Figure um Speicher zu sparen #prevent memory leaks when generating many visualization images
        #plt.show()

def process_all_data():
    """Hauptfunktion zur Verarbeitung aller Daten"""
    processor = TischtennisDataProcessor()
    
    #Ordner für verarbeitete Rohdaten erstellen falls nicht vorhanden
    os.makedirs('./processed_data', exist_ok=True)
    
    stroke_types = ['vorhand_topspin', 'vorhand_schupf', 'vorhand_block', 'rueckhand_topspin', 'rueckhand_schupf', 'rueckhand_block']
    label_map = {stroke: idx for idx, stroke in enumerate(stroke_types)}
    """
    label_map = {}
    for idx, stroke in enumerate(stroke_types):
        label_map[stroke] = idx
    
    # label_map sieht so aus:
    label_map = {
        'vorhand_topspin': 0,
        'vorhand_schupf': 1,
        'vorhand_block': 2,
        'rueckhand_topspin': 3,
        'rueckhand_schupf': 4,
        'rueckhand_block': 5
    }
    """

    all_data = []
    all_labels = []
    
    # Verarbeite jeden Schlagtyp
    for stroke_type in stroke_types:
        windows, features = processor.process_stroke_type(stroke_type)
        
        # Labels hinzufügen
        labels = [label_map[stroke_type]] * len(windows)
        """ 
        labels = [0] * 5    # Beispiel: 5 Schläge => len(windows) = 5 
        # Ergebnis: labels = [0, 0, 0, 0, 0]
        """
        
        all_data.extend(windows)    #Windows von jedem Schlagart wird hier gesammelt 
        all_labels.extend(labels)   #Label von alle Daten: 0,1,2,3
    
    # In numpy arrays konvertieren
    if len(all_data) > 0:
        X = np.array(all_data) 
        y = np.array(all_labels)
        
        print(f"\nGesamtdaten: {X.shape[0]} Schläge")
        print(f"Form: {X.shape}")
        print(f"Features: {len(processor.training_features)} - {processor.training_features}")
        print(f"Klassen: {np.unique(y, return_counts=True)}")
        
        # Speichern
        np.save('./processed_data/X_data.npy', X)
        np.save('./processed_data/y_labels.npy', y)
        
        """ 
        X hat Form: (n_samples, 200, 10) = (Anzahl Schläge, Zeitschritte, Features)
        Reshape macht daraus: (n_samples * 200, 10)
        """
        # Scaler fit und speichern
        X_reshaped = X.reshape(-1, X.shape[-1])     # -1 = automatisch berechnen
        processor.scaler.fit(X_reshaped)            # standard skaliert --> (z-wert normalisiert ?)
        
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
        
        # Feature-Namen speichern
        with open('./processed_data/feature_names.pkl', 'wb') as f:
            pickle.dump(processor.training_features, f) # Speichert das Dictionary (JSON-Objekte) als Binär in Datei (.pkl oder pickel ooder irgendwas) 
        
        print("\nDaten gespeichert in processed_data/")
        
        return X, y
    else:
        print("\nKeine Daten zum Verarbeiten gefunden!")
        return np.array([]), np.array([])

if __name__ == "__main__":
    # Beispiel: Einzelne Datei visualisieren
    processor = TischtennisDataProcessor()
    
    """
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
    """
    
    # Alle Daten verarbeiten
    X, y = process_all_data()
