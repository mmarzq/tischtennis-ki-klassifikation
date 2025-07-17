"""
MINIMALE Datenvorverarbeitung für Tischtennisschlag-Klassifikation
Mit JSON statt Pickle für bessere Lesbarkeit und Sicherheit
Nur mit pandas, numpy und json (Python Standard-Bibliotheken)

Simpler gemacht:
    - Butterworth-Filter → simple_smooth() (einfacher gleitender Durchschnitt)
    - scipy.find_peaks → find_movement_peaks() (eigene Peak-Erkennung)
    - sklearn.StandardScaler → simple_normalize() (manuelle Z-Score Normalisierung)
    - pickle → JSON für alle Metadaten (NumPy Arrays bleiben als .npy)
"""

import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt 

class MinimalTischtennisProcessor:
    def __init__(self, window_size=200):
        self.window_size = window_size
        
        # Features die wir verwenden
        self.training_features = [
            'gyro_x', 'gyro_y', 'gyro_z',
            'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
            'quat_w', 'quat_x', 'quat_y', 'quat_z'
        ]
        
        # Für einfache Normalisierung (statt sklearn StandardScaler)
        self.feature_means = None
        self.feature_stds = None
    
    def load_csv_data(self, filepath):
        """Lädt CSV-Datei und normalisiert Spaltennamen"""
        try:
            df = pd.read_csv(filepath)
        except:
            print(f"Fehler beim Laden von {filepath}")
            return pd.DataFrame()
        
        # Spaltennamen vereinheitlichen
        column_map = {
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
        
        df.rename(columns=column_map, inplace=True)
        
        # Fehlende Spalten mit 0 auffüllen
        for feature in self.training_features:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Nur benötigte Spalten behalten
        needed_cols = ['timestamp'] + self.training_features
        available_cols = [col for col in needed_cols if col in df.columns]
        
        return df[available_cols]
    
    def simple_smooth(self, values, window=5):
        """
        Einfache Glättung - ersetzt komplizierte Filter
        Einfacher gleitender Durchschnitt
        """
        if len(values) < window:
            return values
        
        smoothed = np.zeros_like(values)
        half_window = window // 2
        
        # Anfang und Ende einfach kopieren
        smoothed[:half_window] = values[:half_window]
        smoothed[-half_window:] = values[-half_window:]
        
        # Mittlerer Teil - gleitender Durchschnitt
        for i in range(half_window, len(values) - half_window):
            smoothed[i] = np.mean(values[i-half_window:i+half_window+1])
        
        return smoothed
    
    def find_movement_peaks(self, data):
        """
        Einfache Schlagerkennung - ersetzt scipy find_peaks
        Findet Bewegungsintensitäts-Spitzen
        """
        # Bewegungsintensität berechnen
        lin_acc = np.sqrt(data['lin_acc_x']**2 + data['lin_acc_y']**2 + data['lin_acc_z']**2)
        gyro = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)
        
        # Einfache Normalisierung
        lin_acc_norm = (lin_acc - np.mean(lin_acc)) / (np.std(lin_acc) + 0.001)
        gyro_norm = (gyro - np.mean(gyro)) / (np.std(gyro) + 0.001)
        
        # Bewegungsintensität (70% Beschleunigung, 30% Gyro)
        intensity = 0.7 * lin_acc_norm + 0.3 * gyro_norm
        
        # Glättung
        intensity_smooth = self.simple_smooth(intensity, window=5)
        
        # Schwellwert für Peaks
        threshold = np.mean(intensity_smooth) + 0.5 * np.std(intensity_smooth)
        
        # Einfache Peak-Suche
        peaks = []
        min_distance = 50  # Mindestabstand zwischen Schlägen
        
        for i in range(1, len(intensity_smooth) - 1):
            # Ist es ein lokales Maximum über dem Schwellwert?
            if (intensity_smooth[i] > threshold and 
                intensity_smooth[i] > intensity_smooth[i-1] and 
                intensity_smooth[i] > intensity_smooth[i+1]):
                
                # Prüfe Mindestabstand zu vorherigen Peaks
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return peaks, intensity_smooth
    
    def extract_windows_around_peaks(self, data, peaks):
        """Schneidet Fenster um erkannte Peaks heraus"""
        windows = []
        half_window = self.window_size // 2
        
        for peak in peaks:
            start = max(0, peak - half_window)
            end = min(len(data), peak + half_window)
            
            # Nur vollständige Fenster verwenden
            if end - start == self.window_size:
                window_data = data.iloc[start:end][self.training_features].values
                windows.append(window_data)
        
        return windows
    
    def process_one_stroke_type(self, stroke_type, data_folder):
        """Verarbeitet alle Dateien eines Schlagtyps"""
        pattern = f"{data_folder}/{stroke_type}/*.csv"
        files = glob.glob(pattern)
        
        print(f"Verarbeite {stroke_type}: {len(files)} Dateien")
        
        all_windows = []
        
        # Ordner für Visualisierungen erstellen
        visual_processed_data_folder = os.path.join('processed_data', 'visual_processed', stroke_type)
        os.makedirs(visual_processed_data_folder, exist_ok=True)
        
        for file in files:
            data = self.load_csv_data(file)
            
            if len(data) < self.window_size:
                print(f"  {os.path.basename(file)}: Zu wenig Daten ({len(data)} Samples)")
                continue
            
            # Einfache Glättung auf Sensordaten
            for feature in ['gyro_x', 'gyro_y', 'gyro_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']:
                if feature in data.columns:
                    data[feature] = self.simple_smooth(data[feature].values)
            
            # Peaks finden
            peaks, intensity = self.find_movement_peaks(data)
            
            # Visualisiere die Schlagerkennung
            filename = os.path.basename(file)
            visualization_path = os.path.join(visual_processed_data_folder, f"{filename}.png")
            self.visualize_stroke_detection(data, peaks, intensity, visualization_path)
            
            # Fenster extrahieren
            windows = self.extract_windows_around_peaks(data, peaks)
            all_windows.extend(windows)
            
            # Visualisiere jedes extrahierte Fenster
            for i, window in enumerate(windows):
                window_df = pd.DataFrame(window, columns=self.training_features)
                window_name = os.path.join(visual_processed_data_folder, f"{filename}_window_{i}.png")
                self.visualize_stroke_detection(window_df, output_file=window_name)
            
            print(f"  {os.path.basename(file)}: {len(windows)} Schläge erkannt")
        
        return all_windows
    
    def simple_normalize(self, X):
        """
        Einfache Normalisierung - ersetzt sklearn StandardScaler
        Z-Score Normalisierung: (x - mean) / std
        """
        # X hat Form: (n_samples, window_size, n_features)
        X_reshaped = X.reshape(-1, X.shape[-1])  # Alle Zeitpunkte zusammen
        
        # Mittelwert und Standardabweichung berechnen
        if self.feature_means is None:
            self.feature_means = np.mean(X_reshaped, axis=0)
            self.feature_stds = np.std(X_reshaped, axis=0) + 1e-8  # Kleine Zahl um Division durch 0 zu vermeiden
        
        # Normalisierung anwenden
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_normalized[i, j] = (X[i, j] - self.feature_means) / self.feature_stds
        
        return X_normalized
    
    def save_normalization_params(self, filename):
        """Speichert Normalisierungsparameter als JSON"""
        params = {
            'means': self.feature_means.tolist(),  # NumPy Array zu Liste
            'stds': self.feature_stds.tolist(),
            'features': self.training_features
        }
        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)  # indent für bessere Lesbarkeit
        print(f"Normalisierungsparameter gespeichert in {filename}")
    
    def load_normalization_params(self, filename):
        """Lädt Normalisierungsparameter aus JSON"""
        with open(filename, 'r') as f:
            params = json.load(f)
        self.feature_means = np.array(params['means'])  # Liste zu NumPy Array
        self.feature_stds = np.array(params['stds'])
        return params['features']
    
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
            plt.close()  # Schließe Figure um Speicher zu sparen

def process_all_data_minimal():
    """
    Hauptfunktion - verarbeitet alle Daten mit minimalen Bibliotheken
    """
    processor = MinimalTischtennisProcessor(window_size=100) #modifiziert: Fenstergröße 200
    
    # Schlagtypen definieren
    stroke_types = ['vorhand_topspin', 'vorhand_schupf', 'rueckhand_topspin', 'rueckhand_schupf']
    
    all_data = []
    all_labels = []
    
    # Jeder Schlagtyp bekommt eine Nummer als Label
    for label_num, stroke_type in enumerate(stroke_types):
        windows = processor.process_one_stroke_type(stroke_type, './rohdaten')
        
        if windows:
            all_data.extend(windows)
            # Label für jeden Schlag dieses Typs
            all_labels.extend([label_num] * len(windows))
    
    if not all_data:
        print("Keine Daten gefunden!")
        return None, None
    
    # In numpy arrays umwandeln
    X = np.array(all_data)  # Form: (n_samples, window_size, n_features)
    y = np.array(all_labels)  # Form: (n_samples,)
    
    print(f"\nErgebnis:")
    print(f"Anzahl Schläge: {len(X)}")
    print(f"Datenform: {X.shape}")
    print(f"Features: {len(processor.training_features)}")
    print(f"Schlagtypen: {stroke_types}")
    
    # Daten normalisieren
    X_normalized = processor.simple_normalize(X)
    
    # Ausgabeordner erstellen
    output_dir = './processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # NumPy Arrays speichern (bleiben als .npy für Effizienz)
    np.save(f'{output_dir}/X_minimal.npy', X_normalized)
    np.save(f'{output_dir}/y_minimal.npy', y)
    
    # Normalisierungsparameter als JSON speichern
    processor.save_normalization_params(f'{output_dir}/normalization_minimal.json')
    
    # Zusätzliche Info als JSON speichern
    info = {
        'stroke_types': stroke_types,
        'window_size': processor.window_size,
        'features': processor.training_features,
        'data_shape': list(X.shape),  # Tuple zu Liste für JSON
        'num_samples': int(X.shape[0]),
        'num_features': int(X.shape[2]),
        'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f'{output_dir}/info_minimal.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Zusammenfassung der Klassen als JSON
    unique_labels, counts = np.unique(y, return_counts=True)
    class_distribution = {
        stroke_types[int(label)]: int(count) 
        for label, count in zip(unique_labels, counts)
    }
    
    summary = {
        'total_samples': int(len(X)),
        'class_distribution': class_distribution,
        'balanced': bool(np.std(counts) < np.mean(counts) * 0.2)  # Prüft ob Klassen ausbalanciert sind
    }
    
    with open(f'{output_dir}/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDaten gespeichert in {output_dir}/")
    print("- X_minimal.npy (normalisierte Sensordaten)")
    print("- y_minimal.npy (Labels)")
    print("- normalization_minimal.json (Normalisierungsparameter für Features)")
    print("- info_minimal.json (Zusätzliche Informationen über die Daten)")
    print("- data_summary.json (Zusammenfassung der Daten und Klassenverteilung)")
    
    return X_normalized, y

# Einfache Funktion um die Ergebnisse zu prüfen
def check_processed_data():
    """Lädt und zeigt verarbeitete Daten"""
    try:
        X = np.load('./processed_data/X_minimal.npy')
        y = np.load('./processed_data/y_minimal.npy')
        
        # JSON Dateien laden
        with open('./processed_data/info_minimal.json', 'r') as f:
            info = json.load(f)
        
        with open('./processed_data/data_summary.json', 'r') as f:
            summary = json.load(f)
        
        print("Verarbeitete Daten geladen:")
        print(f"X Form: {X.shape}")
        print(f"y Form: {y.shape}")
        print(f"Schlagtypen: {info['stroke_types']}")
        print(f"Features: {info['features']}")
        print(f"\nKlassenverteilung:")
        for stroke_type, count in summary['class_distribution'].items():
            print(f"  {stroke_type}: {count} Schläge")
        print(f"\nDaten ausbalanciert: {'Ja' if summary['balanced'] else 'Nein'}")
        
    except FileNotFoundError as e:
        print(f"Datei nicht gefunden: {e}")
        print("Noch keine verarbeiteten Daten gefunden. Führen Sie zuerst process_all_data_minimal() aus.")

if __name__ == "__main__":
    print("=== MINIMALE DATENVERARBEITUNG MIT JSON ===")
    print("Verwendet: pandas, numpy, json")
    print("JSON-Dateien sind lesbar und sicher!")
    print()
    
    # Alle Daten verarbeiten
    X, y = process_all_data_minimal()
    
    if X is not None:
        print("\n=== ERFOLGREICH VERARBEITET ===")
        print("Die Daten sind jetzt bereit für das Training!")
        print("Öffnen Sie die JSON-Dateien mit einem Texteditor um die Metadaten zu sehen.")
    else:
        print("\n=== FEHLER ===")
        print("Keine Daten konnten verarbeitet werden.")