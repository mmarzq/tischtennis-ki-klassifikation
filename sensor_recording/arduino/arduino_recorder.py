"""
Arduino Sensor-Datenerfassung für Tischtennisschlag-Klassifikation
Erfasst Daten vom Arduino Nicla Sense ME über serielle Verbindung
"""

import serial
import csv
import time
import os
from datetime import datetime
import pandas as pd
import numpy as np

class ArduinoDataRecorder:
    def __init__(self, port='COM4', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.recording = False
        
    def connect(self):
        """Verbindung zum Arduino herstellen"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino Reset abwarten
            print(f"Verbunden mit {self.port} bei {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Fehler beim Verbinden: {e}")
            return False
    
    def disconnect(self):
        """Verbindung trennen"""
        if self.ser:
            self.ser.close()
            print("Verbindung getrennt")
    
    def record_strokes(self, stroke_type, output_folder, duration=60):
        """
        Nimmt Sensordaten für einen bestimmten Schlagtyp auf
        
        Args:
            stroke_type: Art des Schlags (z.B. 'vorhand_topspin')
            output_folder: Ausgabeordner für CSV-Dateien
            duration: Aufnahmedauer in Sekunden
        """
        if not self.ser:
            print("Keine Verbindung zum Arduino!")
            return
        
        # Zeitstempel für Dateiname
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/{stroke_type}_{timestamp}.csv"
        
        print(f"\nAufnahme von {stroke_type} startet in 3 Sekunden...")
        time.sleep(3)
        
        print(f"Aufnahme läuft für {duration} Sekunden...")
        print("Führen Sie jetzt die Schläge aus!")
        
        data_buffer = []
        start_time = time.time()
        
        # Header für CSV
        headers = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        while time.time() - start_time < duration:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                    if line and ',' in line:
                        values = line.split(',')
                        if len(values) >= 7:  # Timestamp + 6 Sensorwerte
                            data_buffer.append(values[:7])
                except Exception as e:
                    print(f"Fehler beim Lesen: {e}")
        
        # Daten speichern
        if data_buffer:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(data_buffer)
            
            print(f"\nAufnahme beendet. {len(data_buffer)} Datenpunkte gespeichert in:")
            print(f"{filename}")
        else:
            print("\nKeine Daten empfangen!")
        
        return filename
    
    def interactive_recording_session(self):
        """Interaktive Aufnahmesession für alle Schlagtypen"""
        stroke_types = {
            '1': 'vorhand_topspin',
            '2': 'vorhand_schupf',
            '3': 'rueckhand_topspin',
            '4': 'rueckhand_schupf'
        }
        
        base_folder = "../rohdaten"
        
        while True:
            print("\n=== Tischtennisschlag Aufnahme ===")
            print("1: Vorhand Topspin")
            print("2: Vorhand Schupf")
            print("3: Rückhand Topspin")
            print("4: Rückhand Schupf")
            print("q: Beenden")
            
            choice = input("\nWählen Sie einen Schlagtyp: ")
            
            if choice == 'q':
                break
            
            if choice in stroke_types:
                stroke_type = stroke_types[choice]
                output_folder = f"{base_folder}/{stroke_type}"
                
                # Aufnahmedauer festlegen
                duration = input("Aufnahmedauer in Sekunden (Standard: 30): ")
                duration = int(duration) if duration else 30
                
                # Aufnahme starten
                self.record_strokes(stroke_type, output_folder, duration)
                
                # Nochmal aufnehmen?
                again = input("\nNoch eine Aufnahme vom gleichen Typ? (j/n): ")
                if again.lower() == 'j':
                    self.record_strokes(stroke_type, output_folder, duration)

def main():
    # Arduino-Recorder initialisieren
    recorder = ArduinoDataRecorder(port='COM4', baudrate=115200)
    
    # Verbindung herstellen
    if recorder.connect():
        try:
            # Interaktive Aufnahmesession starten
            recorder.interactive_recording_session()
        finally:
            recorder.disconnect()
    else:
        print("Konnte keine Verbindung zum Arduino herstellen!")
        print("Überprüfen Sie den COM-Port und stellen Sie sicher, dass der Arduino-Sketch läuft.")

if __name__ == "__main__":
    main()
