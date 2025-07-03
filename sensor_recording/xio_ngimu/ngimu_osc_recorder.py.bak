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
        self.quaternion_data = {'quat_w': 0, 'quat_x': 0, 'quat_y': 0, 'quat_z': 0}
        
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
            
            data_point = {
                'timestamp': timestamp,
                'gyro_x': gyro[0],
                'gyro_y': gyro[1],
                'gyro_z': gyro[2],
                'acc_x': acc[0],
                'acc_y': acc[1],
                'acc_z': acc[2],
                # Füge aktuelle Quaternion-Daten hinzu
                'quat_w': self.quaternion_data['quat_w'],
                'quat_x': self.quaternion_data['quat_x'],
                'quat_y': self.quaternion_data['quat_y'],
                'quat_z': self.quaternion_data['quat_z']
            }
            
            self.data_queue.put(('sensors', data_point))
            self.current_data.update(data_point)
    
    def handle_quaternion(self, address, *args):
        """Verarbeitet Quaternion-Daten für Orientierung"""
        if len(args) >= 5:  # timestamp + 4 quaternion values
            # Aktualisiere die Quaternion-Daten
            self.quaternion_data['quat_w'] = args[1]
            self.quaternion_data['quat_x'] = args[2]
            self.quaternion_data['quat_y'] = args[3]
            self.quaternion_data['quat_z'] = args[4]
            self.current_data.update(self.quaternion_data)
    
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
        #print("3...")
        #time.sleep(1)
        print("2...")
        time.sleep(1)  
        print("1...")
        time.sleep(1)
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
            headers = ['Timestamp', 'Acc_X', 'Acc_Y', 'Acc_Z', 
                      'Gyro_X', 'Gyro_Y', 'Gyro_Z', 
                      'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']
            
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                
                for data in data_buffer:
                    row = [
                        data.get('timestamp', 0),
                        data.get('acc_x', 0),
                        data.get('acc_y', 0),
                        data.get('acc_z', 0),
                        data.get('gyro_x', 0),
                        data.get('gyro_y', 0),
                        data.get('gyro_z', 0),
                        data.get('quat_w', 0),
                        data.get('quat_x', 0),
                        data.get('quat_y', 0),
                        data.get('quat_z', 0)
                    ]
                    writer.writerow(row)
            
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
        print("\nStellen Sie sicher, dass:")
        print("1. Sie mit dem NGIMU WiFi verbunden sind!")
        print("2. NGIMU Send Rates aktiviert sind:")
        print("   - Sensors: 100Hz")
        print("   - Quaternion: 50-100Hz")
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
            print("4. Send Rates sind aktiviert:")
            print("   - Sensors: 100Hz")
            print("   - Quaternion: 50-100Hz")
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
                
                # Ordner erstellen falls nicht vorhanden
                #os.makedirs(output_folder, exist_ok=True)
                
                # Aufnahmedauer festlegen
                duration = input("Aufnahmedauer in Sekunden (Standard: 30): ")
                duration = int(duration) if duration else 30
                
                # Aufnahme starten
                self.record_strokes(stroke_type, output_folder, duration)
                
                # Nochmal weiter aufnehmen?
                while True:
                    again = input("\nNoch eine Aufnahme vom gleichen Typ? (j/n): ")
                    if again.lower() == 'n':
                        break
                    self.record_strokes(stroke_type, output_folder, duration)

def main():
    """Hauptfunktion"""
    # NGIMU Receiver initialisieren
    # Port 8001 basierend auf Ihrer Konfiguration
    #receiver = NGIMUReceiver(ip="192.168.1.2", port=8002)
    # 0.0.0.0 bedeutet, dass der Server auf allen verfügbaren Netzwerkinterfaces lauscht, einschließlich der IP 192.168.1.2.
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
    print("3. Send Rates aktiviert sind:")
    print("   - Sensors: 100Hz")
    print("   - Quaternion: 100Hz")
    
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
