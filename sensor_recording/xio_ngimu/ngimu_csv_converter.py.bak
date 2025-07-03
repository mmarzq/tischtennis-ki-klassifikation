"""
NGIMU CSV Konverter
Konvertiert NGIMU GUI exportierte CSV-Dateien in das einheitliche Format
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

class NGIMUCSVConverter:
    def __init__(self):
        self.ngimu_columns = {
            # NGIMU CSV Spalten -> Einheitliche Spalten
            'Time (s)': 'timestamp',
            'Gyroscope X (deg/s)': 'gyro_x',
            'Gyroscope Y (deg/s)': 'gyro_y', 
            'Gyroscope Z (deg/s)': 'gyro_z',
            'Accelerometer X (g)': 'acc_x',
            'Accelerometer Y (g)': 'acc_y',
            'Accelerometer Z (g)': 'acc_z',
            'Magnetometer X (uT)': 'mag_x',
            'Magnetometer Y (uT)': 'mag_y',
            'Magnetometer Z (uT)': 'mag_z'
        }
        
    def convert_ngimu_csv(self, input_file, output_file=None):
        """Konvertiert eine NGIMU CSV-Datei"""
        print(f"Konvertiere: {input_file}")
        
        # CSV laden
        df = pd.read_csv(input_file)
        
        # Spalten umbenennen
        df_renamed = pd.DataFrame()
        
        for ngimu_col, unified_col in self.ngimu_columns.items():
            if ngimu_col in df.columns:
                df_renamed[unified_col] = df[ngimu_col]
            else:
                # Alternative Spaltennamen probieren
                for col in df.columns:
                    if unified_col in col.lower():
                        df_renamed[unified_col] = df[col]
                        break
        
        # Timestamp in Millisekunden umrechnen wenn nötig
        if 'timestamp' in df_renamed.columns:
            df_renamed['timestamp'] = (df_renamed['timestamp'] * 1000).astype(int)
        
        # Fehlende Spalten mit Nullen füllen
        required_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 
                        'gyro_x', 'gyro_y', 'gyro_z']
        for col in required_cols:
            if col not in df_renamed.columns:
                df_renamed[col] = 0
        
        # Spalten in richtiger Reihenfolge
        final_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 
                     'gyro_x', 'gyro_y', 'gyro_z']
        if all(col in df_renamed.columns for col in ['mag_x', 'mag_y', 'mag_z']):
            final_cols.extend(['mag_x', 'mag_y', 'mag_z'])
        
        df_final = df_renamed[final_cols]
        
        # Ausgabedatei bestimmen
        if output_file is None:
            base_name = os.path.basename(input_file)
            output_file = base_name.replace('.csv', '_converted.csv')
        
        # Speichern
        df_final.to_csv(output_file, index=False)
        print(f"Gespeichert als: {output_file}")
        
        return df_final
    
    def batch_convert_folder(self, input_folder, stroke_type):
        """Konvertiert alle CSV-Dateien in einem Ordner"""
        output_folder = f"../../rohdaten/{stroke_type}"
        os.makedirs(output_folder, exist_ok=True)
        
        # Alle CSV-Dateien finden
        csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
        
        if not csv_files:
            print(f"Keine CSV-Dateien in {input_folder} gefunden!")
            return
        
        print(f"\nKonvertiere {len(csv_files)} Dateien für {stroke_type}...")
        
        for csv_file in csv_files:
            # Neuer Dateiname mit Zeitstempel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(csv_file).replace('.csv', '')
            output_file = os.path.join(output_folder, 
                                     f"{stroke_type}_{base_name}_{timestamp}.csv")
            
            try:
                self.convert_ngimu_csv(csv_file, output_file)
            except Exception as e:
                print(f"Fehler bei {csv_file}: {e}")
    
    def analyze_ngimu_csv(self, csv_file):
        """Analysiert eine NGIMU CSV-Datei und zeigt Informationen"""
        df = pd.read_csv(csv_file)
        
        print(f"\n=== Analyse von {os.path.basename(csv_file)} ===")
        print(f"Zeilen: {len(df)}")
        print(f"Spalten: {len(df.columns)}")
        print(f"\nVerfügbare Spalten:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}: {col}")
        
        # Datenrate schätzen
        if 'Time (s)' in df.columns:
            time_diff = df['Time (s)'].diff().dropna()
            avg_period = time_diff.mean()
            estimated_rate = 1 / avg_period if avg_period > 0 else 0
            print(f"\nGeschätzte Datenrate: {estimated_rate:.1f} Hz")
            print(f"Aufnahmedauer: {df['Time (s)'].iloc[-1]:.1f} Sekunden")
        
        # Basis-Statistiken für Sensoren
        sensor_cols = [col for col in df.columns if any(
            sensor in col for sensor in ['Gyroscope', 'Accelerometer', 'Magnetometer']
        )]
        
        if sensor_cols:
            print(f"\nSensor-Statistiken:")
            for col in sensor_cols[:6]:  # Nur erste 6 anzeigen
                values = df[col]
                print(f"  {col}:")
                print(f"    Min: {values.min():.2f}, Max: {values.max():.2f}, "
                     f"Mittel: {values.mean():.2f}, Std: {values.std():.2f}")

def interactive_converter():
    """Interaktive Konvertierungssession"""
    converter = NGIMUCSVConverter()
    
    while True:
        print("\n=== NGIMU CSV Konverter ===")
        print("1: Einzelne Datei konvertieren")
        print("2: Ordner batch-konvertieren")
        print("3: CSV-Datei analysieren")
        print("q: Beenden")
        
        choice = input("\nWählen Sie eine Option: ")
        
        if choice == 'q':
            break
        elif choice == '1':
            file_path = input("Pfad zur CSV-Datei: ").strip('"')
            if os.path.exists(file_path):
                converter.convert_ngimu_csv(file_path)
            else:
                print("Datei nicht gefunden!")
        elif choice == '2':
            folder_path = input("Pfad zum Ordner mit CSV-Dateien: ").strip('"')
            if os.path.exists(folder_path):
                print("\nFür welchen Schlagtyp?")
                print("1: vorhand_topspin")
                print("2: vorhand_schupf")
                print("3: rueckhand_topspin")
                print("4: rueckhand_schupf")
                
                stroke_map = {
                    '1': 'vorhand_topspin',
                    '2': 'vorhand_schupf',
                    '3': 'rueckhand_topspin',
                    '4': 'rueckhand_schupf'
                }
                
                stroke_choice = input("Wahl: ")
                if stroke_choice in stroke_map:
                    converter.batch_convert_folder(folder_path, stroke_map[stroke_choice])
            else:
                print("Ordner nicht gefunden!")
        elif choice == '3':
            file_path = input("Pfad zur CSV-Datei: ").strip('"')
            if os.path.exists(file_path):
                converter.analyze_ngimu_csv(file_path)
            else:
                print("Datei nicht gefunden!")

def main():
    """Hauptfunktion"""
    import sys
    
    if len(sys.argv) > 1:
        # Kommandozeilen-Modus
        converter = NGIMUCSVConverter()
        for file_path in sys.argv[1:]:
            if os.path.exists(file_path):
                converter.convert_ngimu_csv(file_path)
    else:
        # Interaktiver Modus
        interactive_converter()

if __name__ == "__main__":
    main()
