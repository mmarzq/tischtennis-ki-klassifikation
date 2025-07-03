import csv
from datetime import datetime
import os
import time
import osc_decoder
import socket

# Konstanten
RECORDING_DURATION = 30  # Sekunden  
NGIMU_IP = "192.168.1.1"  # IP-Adresse des NGIMU
NGIMU_SEND_PORT = 9000  # Port für Befehle an NGIMU
RECEIVE_PORTS = [8001, 8010, 8011, 8012]  # Ports für Empfang von Daten

# CSV Header 
HEADER = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 
          'Mag_X', 'Mag_Y', 'Mag_Z', 'Bar', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']

# Globale Variablen für das Zusammenführen der Sensordaten
current_sensors_data = {}
current_quaternion_data = {}


def merge_and_write_data(csv_writer, timestamp):
    """
    Fügt Sensor- und Quaternion-Daten zusammen und schreibt sie in die CSV-Datei
    """
    if current_sensors_data and current_quaternion_data:
        row = [
            timestamp,
            current_sensors_data.get('gyro_x', 0.0),
            current_sensors_data.get('gyro_y', 0.0), 
            current_sensors_data.get('gyro_z', 0.0),
            current_sensors_data.get('acc_x', 0.0),
            current_sensors_data.get('acc_y', 0.0),
            current_sensors_data.get('acc_z', 0.0),
            current_sensors_data.get('mag_x', 0.0),
            current_sensors_data.get('mag_y', 0.0),
            current_sensors_data.get('mag_z', 0.0),
            current_sensors_data.get('baro', 0.0),
            current_quaternion_data.get('w', 0.0),
            current_quaternion_data.get('x', 0.0),
            current_quaternion_data.get('y', 0.0),
            current_quaternion_data.get('z', 0.0)
        ]
        csv_writer.writerow(row)
        return True
    return False


def setup_sockets():
    """
    Erstellt und konfiguriert die Socket-Verbindungen
    """
    # Socket für Senden erstellen
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Sockets für Empfang erstellen
    receive_sockets = []
    for port in RECEIVE_PORTS:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", port))
        sock.setblocking(False)
        receive_sockets.append(sock)
    
    return send_socket, receive_sockets


def record_data(duration=RECORDING_DURATION):
    """
    Hauptfunktion für die Datenaufnahme
    """
    global current_sensors_data, current_quaternion_data
    
    # Sockets einrichten
    send_socket, receive_sockets = setup_sockets()
    
    # Send /identify message an NGIMU
    send_socket.sendto(bytes("/identify\0\0\0,\0\0\0", "utf-8"), (NGIMU_IP, NGIMU_SEND_PORT))
    
    # Data Verzeichnis erstellen
    os.makedirs('data', exist_ok=True)
    
    # CSV-Datei erstellen
    filename = f"data/ngimu_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Starte Aufzeichnung für {duration} Sekunden...")
    print(f"Datei: {filename}")
    
    start_time = time.time()
    data_count = 0
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(HEADER)
            
            while time.time() - start_time < duration:
                # Daten von allen Sockets empfangen
                for udp_socket in receive_sockets:
                    try:
                        data, addr = udp_socket.recvfrom(2048)
                    except socket.error:
                        pass
                    else:
                        # OSC Messages dekodieren
                        for message in osc_decoder.decode(data):
                            if len(message) >= 2:
                                current_timestamp = time.time() - start_time
                                osc_address = message[1]
                                
                                # Sensordaten verarbeiten
                                if osc_address == '/sensors' and len(message) >= 12:
                                    current_sensors_data = {
                                        'gyro_x': message[2],
                                        'gyro_y': message[3], 
                                        'gyro_z': message[4],
                                        'acc_x': message[5],
                                        'acc_y': message[6],
                                        'acc_z': message[7],
                                        'mag_x': message[8],
                                        'mag_y': message[9],
                                        'mag_z': message[10],
                                        'baro': message[11]
                                    }
                                    
                                # Quaternion-Daten verarbeiten  
                                elif osc_address == '/quaternion' and len(message) >= 6:
                                    current_quaternion_data = {
                                        'w': message[2],
                                        'x': message[3],
                                        'y': message[4], 
                                        'z': message[5]
                                    }
                                    
                                    # Daten zusammenführen und schreiben
                                    if merge_and_write_data(csv_writer, current_timestamp):
                                        data_count += 1
                
                # Kurze Pause um CPU zu schonen
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nAufzeichnung durch Benutzer unterbrochen.")
    
    finally:
        # Sockets schließen
        for socket_obj in receive_sockets:
            socket_obj.close()
        send_socket.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nAufzeichnung beendet!")
    print(f"Dauer: {elapsed_time:.1f} Sekunden")
    print(f"Datensätze: {data_count}")
    print(f"Rate: {data_count/elapsed_time:.1f} Hz")
    
    return filename


if __name__ == "__main__":
    # Benutzer nach Aufnahmedauer fragen
    try:
        duration = input(f"Aufnahmedauer in Sekunden (Standard: {RECORDING_DURATION}): ")
        if duration:
            duration = int(duration)
        else:
            duration = RECORDING_DURATION
    except ValueError:
        print(f"Ungültige Eingabe, verwende Standard: {RECORDING_DURATION} Sekunden")
        duration = RECORDING_DURATION
    
    # Aufnahme starten
    record_data(duration)
