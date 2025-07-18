"""
Sensor-Vergleichstool
Vergleicht Daten von Arduino Nicla Sense ME und X-IO NGIMU
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import os

class SensorComparison:
    def __init__(self):
        self.arduino_data = None
        self.ngimu_data = None
        
    def load_arduino_data(self, filepath):
        """Lädt Arduino-Daten"""
        self.arduino_data = pd.read_csv(filepath)
        # Zeitstempel normalisieren (ms zu Sekunden)
        if 'timestamp' in self.arduino_data.columns:
            self.arduino_data['time'] = (self.arduino_data['timestamp'] - 
                                        self.arduino_data['timestamp'].iloc[0]) / 1000
        print(f"Arduino-Daten geladen: {len(self.arduino_data)} Samples")
        
    def load_ngimu_data(self, filepath):
        """Lädt NGIMU-Daten"""
        self.ngimu_data = pd.read_csv(filepath)
        # Zeitstempel normalisieren
        if 'timestamp' in self.ngimu_data.columns:
            self.ngimu_data['time'] = (self.ngimu_data['timestamp'] - 
                                       self.ngimu_data['timestamp'].iloc[0]) / 1000
        print(f"NGIMU-Daten geladen: {len(self.ngimu_data)} Samples")
    
    def calculate_sampling_rates(self):
        """Berechnet die tatsächlichen Abtastraten"""
        results = {}
        
        if self.arduino_data is not None and 'time' in self.arduino_data.columns:
            time_diff = self.arduino_data['time'].diff().dropna()
            arduino_rate = 1 / time_diff.mean() if time_diff.mean() > 0 else 0
            results['Arduino'] = {
                'rate': arduino_rate,
                'std': 1 / time_diff.std() if time_diff.std() > 0 else 0,
                'duration': self.arduino_data['time'].iloc[-1]
            }
        
        if self.ngimu_data is not None and 'time' in self.ngimu_data.columns:
            time_diff = self.ngimu_data['time'].diff().dropna()
            ngimu_rate = 1 / time_diff.mean() if time_diff.mean() > 0 else 0
            results['NGIMU'] = {
                'rate': ngimu_rate,
                'std': 1 / time_diff.std() if time_diff.std() > 0 else 0,
                'duration': self.ngimu_data['time'].iloc[-1]
            }
        
        return results
    
    def calculate_noise_levels(self):
        """Berechnet Rauschpegel der Sensoren"""
        results = {}
        
        # Sensoren die verglichen werden sollen
        sensors = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        for sensor_name, data in [('Arduino', self.arduino_data), ('NGIMU', self.ngimu_data)]:
            if data is not None:
                noise_levels = {}
                for sensor in sensors:
                    if sensor in data.columns:
                        # Hochpassfilter für Rauschmessung
                        b, a = signal.butter(4, 0.1, 'high')
                        filtered = signal.filtfilt(b, a, data[sensor])
                        noise_levels[sensor] = np.std(filtered)
                results[sensor_name] = noise_levels
        
        return results
    
    def plot_comparison(self, sensor_type='acc'):
        """Plottet Sensorvergleich"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Sensoren auswählen
        if sensor_type == 'acc':
            sensors = ['acc_x', 'acc_y', 'acc_z']
            unit = 'g'
            title = 'Beschleunigungssensor'
        else:
            sensors = ['gyro_x', 'gyro_y', 'gyro_z']
            unit = '°/s'
            title = 'Gyroskop'
        
        # Erste 3 Subplots für X, Y, Z
        for idx, sensor in enumerate(sensors):
            ax = axes[idx]
            
            if self.arduino_data is not None and sensor in self.arduino_data.columns:
                ax.plot(self.arduino_data['time'], self.arduino_data[sensor], 
                       'b-', label='Arduino', alpha=0.7, linewidth=1.5)
            
            if self.ngimu_data is not None and sensor in self.ngimu_data.columns:
                ax.plot(self.ngimu_data['time'], self.ngimu_data[sensor], 
                       'r-', label='NGIMU', alpha=0.7, linewidth=1.5)
            
            ax.set_ylabel(f'{sensor} ({unit})')
            ax.set_xlabel('Zeit (s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Vierter Subplot für Magnitude
        ax = axes[3]
        
        if self.arduino_data is not None:
            arduino_mag = np.sqrt(
                self.arduino_data[sensors[0]]**2 + 
                self.arduino_data[sensors[1]]**2 + 
                self.arduino_data[sensors[2]]**2
            )
            ax.plot(self.arduino_data['time'], arduino_mag, 
                   'b-', label='Arduino Magnitude', linewidth=2)
        
        if self.ngimu_data is not None:
            ngimu_mag = np.sqrt(
                self.ngimu_data[sensors[0]]**2 + 
                self.ngimu_data[sensors[1]]**2 + 
                self.ngimu_data[sensors[2]]**2
            )
            ax.plot(self.ngimu_data['time'], ngimu_mag, 
                   'r-', label='NGIMU Magnitude', linewidth=2)
        
        ax.set_ylabel(f'Magnitude ({unit})')
        ax.set_xlabel('Zeit (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{title} Vergleich: Arduino vs NGIMU', fontsize=16)
        plt.tight_layout()
        return fig
    
    def generate_comparison_report(self):
        """Erstellt einen detaillierten Vergleichsbericht"""
        print("\n=== SENSOR VERGLEICHSBERICHT ===\n")
        
        # Abtastraten
        rates = self.calculate_sampling_rates()
        print("1. ABTASTRATEN:")
        for sensor, data in rates.items():
            print(f"   {sensor}:")
            print(f"   - Rate: {data['rate']:.1f} Hz (±{data['std']:.1f} Hz)")
            print(f"   - Dauer: {data['duration']:.1f} s")
        
        # Rauschpegel
        noise = self.calculate_noise_levels()
        print("\n2. RAUSCHPEGEL:")
        for sensor_name, levels in noise.items():
            print(f"   {sensor_name}:")
            for axis, level in levels.items():
                print(f"   - {axis}: {level:.4f}")
        
        # Datenqualität
        print("\n3. DATENQUALITÄT:")
        for name, data in [('Arduino', self.arduino_data), ('NGIMU', self.ngimu_data)]:
            if data is not None:
                missing = data.isnull().sum().sum()
                print(f"   {name}:")
                print(f"   - Fehlende Werte: {missing}")
                print(f"   - Datenpunkte: {len(data)}")
        
        # Empfehlungen
        print("\n4. EMPFEHLUNGEN:")
        
        # Basierend auf den Raten
        if 'Arduino' in rates and 'NGIMU' in rates:
            if rates['NGIMU']['rate'] > rates['Arduino']['rate']:
                print("   - NGIMU bietet höhere Abtastrate → Besser für schnelle Bewegungen")
            else:
                print("   - Arduino bietet höhere Abtastrate → Besser für schnelle Bewegungen")
        
        # Basierend auf Rauschen
        if 'Arduino' in noise and 'NGIMU' in noise:
            arduino_noise = np.mean(list(noise['Arduino'].values()))
            ngimu_noise = np.mean(list(noise['NGIMU'].values()))
            
            if ngimu_noise < arduino_noise:
                print("   - NGIMU hat weniger Rauschen → Bessere Signalqualität")
            else:
                print("   - Arduino hat weniger Rauschen → Bessere Signalqualität")
        
        print("\n" + "="*40)

def main():
    """Hauptfunktion für interaktiven Vergleich"""
    comparison = SensorComparison()
    
    print("=== Sensor-Vergleichstool ===")
    print("Vergleicht Arduino Nicla Sense ME mit X-IO NGIMU\n")
    
    # Arduino-Daten laden
    arduino_file = input("Pfad zur Arduino CSV-Datei (oder Enter zum Überspringen): ").strip('"')
    if arduino_file and os.path.exists(arduino_file):
        comparison.load_arduino_data(arduino_file)
    
    # NGIMU-Daten laden
    ngimu_file = input("Pfad zur NGIMU CSV-Datei (oder Enter zum Überspringen): ").strip('"')
    if ngimu_file and os.path.exists(ngimu_file):
        comparison.load_ngimu_data(ngimu_file)
    
    if comparison.arduino_data is None and comparison.ngimu_data is None:
        print("Keine Daten geladen!")
        return
    
    # Bericht generieren
    comparison.generate_comparison_report()
    
    # Plots anzeigen?
    show_plots = input("\nPlots anzeigen? (j/n): ")
    if show_plots.lower() == 'j':
        # Beschleunigung
        fig1 = comparison.plot_comparison('acc')
        plt.savefig('../../visualizations/sensor_comparison_acc.png')
        
        # Gyroskop
        fig2 = comparison.plot_comparison('gyro')
        plt.savefig('../../visualizations/sensor_comparison_gyro.png')
        
        plt.show()

if __name__ == "__main__":
    main()
