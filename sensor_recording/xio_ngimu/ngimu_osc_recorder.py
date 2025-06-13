"""
X-IO NGIMU Datenerfassung über OSC (Open Sound Control)
Empfängt Echtzeit-Sensordaten über WiFi/Ethernet
"""

import time
import csv
import os
from datetime import datetime
from pythonosc import dispatcher, osc_server
import threading
import queue
import numpy as np

class NGIMUReceiver:
    def __init__(self, ip="0.0.0.0", port=8001):  # GEÄNDERT: Port 8001 statt 9000
        self.ip = ip
        self.port = port
        self.data_queue = queue.Queue()
        self.recording = False
        self.current_data = {}
        
        # OSC Dispatcher einrichten
        self.dispatcher = dispatcher.Dispatcher()
        self.setup_handlers()
        
        # Server
        self.server = None
        self.server_thread = None
        
    def setup_handlers(self):
        """OSC-Handler für verschiedene Datentypen einrichten"""
        # Sensoren (100Hz typisch)
        self.dispatcher.map("/sensors", self.handle_sensors)
        
        # Quaternion (50-100Hz)
        self.dispatcher.map("/quaternion", self.handle_quaternion)
        
        # Lineare Beschleunigung (optional)
        self.dispatcher.map("/linear", self.handle_linear)
        
        # Euler-Winkel (optional für Debug)
        self.dispatcher.map("/euler", self.handle_euler)
        
        # Batterie-Status
        self.dispatcher.map("/battery", self.handle_battery)
        
    def handle_sensors(self, address, *args):
        """Verarbeitet Gyro, Acc, Mag Daten"""
        if self.recording and len(args) >= 10:
            # Format: timestamp, gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ
            timestamp = args[0]
            gyro = args[1:4]
            acc = args[4:7]
            mag = args[7:10]
            
            data_point = {
                'timestamp': timestamp,
                'gyro_x': gyro[0],
                'gyro_y': gyro[1],
                'gyro_z': gyro[2],
                'acc_x': acc[0],
                'acc_y': acc[1],
                'acc_z': acc[2],
                'mag_x': mag[0],
                'mag_y': mag[1],
                'mag_z': mag[2]
            }
            
            self.data_queue.put(('sensors', data_point))
            self.current_data.update(data_point)
    
    def handle_quaternion(self, address, *args):
        """Verarbeitet Quaternion-Daten für Orientierung"""
        if len(args) >= 4:
            self.current_data['quat_w'] = args[0]
            self.current_data['quat_x'] = args[1]
            self.current_data['quat_y'] = args[2]
            self.current_data['quat_z'] = args[3]
    
    def handle_linear(self, address, *args):
        """Verarbeitet lineare Beschleunigung (Gravitation entfernt)"""
        if len(args) >= 3:
            self.current_data['lin_acc_x'] = args[0]
            self.current_data['lin_acc_y'] = args[1]
            self.current_data['lin_acc_z'] = args[2]
    
    def handle_euler(self, address, *args):
        """Verarbeitet Euler-Winkel (Roll, Pitch, Yaw)"""
        if len(args) >= 3:
            self.current_data['roll'] = args[0]
            self.current_data['pitch'] = args[1]
            self.current_data['yaw'] = args[2]
    
    def handle_battery(self, address, *args):
        """Batterie-Status"""
        if args:
            print(f"Batterie: {args[0]}%")
    
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
            print("OSC-Server gestoppt")
    
    def record_strokes(self, stroke_type, output_folder, duration=60):
        """
        Nimmt Sensordaten für einen bestimmten Schlagtyp auf
        """
        # Zeitstempel für Dateiname
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/{stroke_type}_{timestamp}_ngimu.csv"
        
        print(f"\nAufnahme von {stroke_type} startet in 3 Sekunden...")
        print("Stellen Sie sicher, dass NGIMU eingeschaltet und verbunden ist!")
        time.sleep(3)
        
        print(f"Aufnahme läuft für {duration} Sekunden...")
        print("Führen Sie jetzt die Schläge aus!")
        
        self.recording = True
        data_buffer = []
        start_time = time.time()
        
        # Daten sammeln
        while time.time() - start_time < duration:
            try:
                # Daten aus Queue holen (non-blocking)
                data_type, data = self.data_queue.get(timeout=0.1)
                if data_type == 'sensors':
                    data_buffer.append(data)
            except queue.Empty:
                continue
        
        self.recording = False
        
        # Daten speichern
        if data_buffer:
            headers = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 
                      'gyro_x', 'gyro_y', 'gyro_z', 
                      'mag_x', 'mag_y', 'mag_z']
            
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(data_buffer)
            
            print(f"\nAufnahme beendet. {len(data_buffer)} Datenpunkte gespeichert in:")
            print(f"{filename}")
        else:
            print("\nKeine Daten empfangen! Überprüfen Sie die NGIMU-Verbindung.")
        
        return filename
    
    def check_connection(self, timeout=5):
        """Überprüft ob NGIMU Daten sendet"""
        print(f"Überprüfe NGIMU-Verbindung für {timeout} Sekunden...")
        print(f"Lausche auf Port {self.port}...")
        self.recording = True
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.data_queue.empty():
                print("✓ NGIMU sendet Daten!")
                self.recording = False
                # Queue leeren
                while not self.data_queue.empty():
                    self.data_queue.get()
                return True
            time.sleep(0.1)
        
        self.recording = False
        print("✗ Keine Daten von NGIMU empfangen!")
        print("\nHINWEIS: Laut Ihrer Konfiguration sendet NGIMU an:")
        print(f"  IP: 192.168.1.2")
        print(f"  Port: 8001")
        print("\nStellen Sie sicher, dass Sie mit dem NGIMU WiFi verbunden sind!")
        return False
    
    def interactive_recording_session(self):
        """Interaktive Aufnahmesession für alle Schlagtypen"""
        stroke_types = {
            '1': 'vorhand_topspin',
            '2': 'vorhand_schupf',
            '3': 'rueckhand_topspin',
            '4': 'rueckhand_schupf'
        }
        
        base_folder = "../../rohdaten"
        
        # Verbindung prüfen
        if not self.check_connection():
            print("\nBitte überprüfen Sie:")
            print("1. Sie sind mit dem NGIMU WiFi verbunden (SSID: 'NGIMU - 0040874A')")
            print("2. Ihre IP im NGIMU Netzwerk ist 192.168.1.2")
            print("3. NGIMU sendet an Port 8001")
            print("4. Send Rates sind aktiviert (sensors: 100Hz)")
            return
        
        while True:
            print("\n=== NGIMU Tischtennisschlag Aufnahme ===")
            print("1: Vorhand Topspin")
            print("2: Vorhand Schupf")
            print("3: Rückhand Topspin")
            print("4: Rückhand Schupf")
            print("c: Verbindung testen")
            print("q: Beenden")
            
            choice = input("\nWählen Sie eine Option: ")
            
            if choice == 'q':
                break
            elif choice == 'c':
                self.check_connection()
            elif choice in stroke_types:
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
    """Hauptfunktion"""
    # NGIMU Receiver initialisieren
    # Port 8001 basierend auf Ihrer Konfiguration
    receiver = NGIMUReceiver(ip="0.0.0.0", port=8001)
    
    print("=== X-IO NGIMU Datenerfassung ===")
    print("\nBasierend auf Ihrer NGIMU-Konfiguration:")
    print("- NGIMU ist im AP-Modus (eigenes WiFi)")
    print("- NGIMU IP: 192.168.1.1")
    print("- Sendet an: 192.168.1.2:8001")
    print("\nBitte stellen Sie sicher, dass:")
    print("1. Sie mit dem NGIMU WiFi verbunden sind")
    print("   SSID: 'NGIMU - 0040874A'")
    print("2. Ihre IP-Adresse 192.168.1.2 ist")
    print("3. Send Rates aktiviert sind (sensors: 100Hz)")
    
    # Frage ob Port geändert werden soll
    custom_port = input("\nPort ändern? (Enter für 8001): ")
    if custom_port:
        receiver.port = int(custom_port)
    
    # Server starten
    receiver.start_server()
    
    try:
        # Interaktive Session
        receiver.interactive_recording_session()
    finally:
        receiver.stop_server()

if __name__ == "__main__":
    main()
