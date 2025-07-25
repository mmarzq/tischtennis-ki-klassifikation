import csv
from datetime import datetime
import os
import time
import osc_decoder
import socket

# Konstanten
RECORDING_DURATION = 5  # Sekunden  
NGIMU_IP = "192.168.1.1"  # IP-Adresse des NGIMU
NGIMU_SEND_PORT = 9000  # Port für Befehle an NGIMU
RECEIVE_PORTS = [8001, 8010, 8011, 8012]  # Ports für Empfang von Daten

# CSV Header 
HEADER = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Mag_X', 'Mag_Y', 'Mag_Z',
           'Bar', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z']

# Globale Variablen für das Zusammenführen der Sensordaten
current_sensors_data = {}
current_quaternion_data = {}
current_linacc_data = {}


def merge_and_write_data(csv_writer, timestamp):
    """
    Fügt Sensor- und Quaternion-Daten zusammen und schreibt sie in die CSV-Datei
    """
    if current_sensors_data and current_quaternion_data and current_linacc_data:
        row = [
            timestamp,
            current_sensors_data.get('gyro_x'),
            current_sensors_data.get('gyro_y'), 
            current_sensors_data.get('gyro_z'),
            current_sensors_data.get('acc_x'),
            current_sensors_data.get('acc_y'),
            current_sensors_data.get('acc_z'),
            current_sensors_data.get('mag_x'),
            current_sensors_data.get('mag_y'),
            current_sensors_data.get('mag_z'),
            current_sensors_data.get('baro'),
            current_quaternion_data.get('w'),
            current_quaternion_data.get('x'),
            current_quaternion_data.get('y'),
            current_quaternion_data.get('z'),
            current_linacc_data.get('lin_acc_x'),
            current_linacc_data.get('lin_acc_y'),
            current_linacc_data.get('lin_acc_z')
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


def record_data(duration, output_folder, stroke_type):
    """
    Hauptfunktion für die Datenaufnahme
    """
    global current_sensors_data, current_quaternion_data, current_linacc_data
    
    # Sockets einrichten
    send_socket, receive_sockets = setup_sockets()
    
    # Send /identify message an NGIMU
    send_socket.sendto(bytes("/identify\0\0\0,\0\0\0", "utf-8"), (NGIMU_IP, NGIMU_SEND_PORT))
    
    # Data Verzeichnis erstellen
    os.makedirs('data', exist_ok=True)
    
    # CSV-Datei erstellen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_folder}/{stroke_type}_{timestamp}_ngimu.csv"

    # Hinweis für die Benutzer
    print(f"Schlagtyp: {stroke_type}")
    print(f"Starte Aufzeichnung für {duration} Sekunden...")
    print("2 ...")
    time.sleep(1)  
    print("1 ...")
    time.sleep(1)
    print("0 ..")

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
                                """
                                RX 03-07-2025 16:55:12.9580Z /sensors, 344.6622f, -62.72688f, 82.53185f, 0.2002593f, -0.6366854f, 0.3851145f, -25.15747f, 25.69919f, -15.08241f, 1018.481f
                                RX 03-07-2025 16:55:12.9580Z /quaternion, -0.03451902f, 0.1571929f, -0.3709079f, 0.9127725f
                                RX 03-07-2025 16:55:12.9580Z /linear, -0.1123103f, 0.02957141f, -0.283576f

                                """
                                
                                # Sensor
                                if osc_address == '/sensors' and len(message) == 12:
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
                                
                                # Quaternion-Daten   
                                elif osc_address == '/quaternion' and len(message) == 6:
                                    current_quaternion_data = {
                                        'w': message[2],
                                        'x': message[3],
                                        'y': message[4], 
                                        'z': message[5]
                                    }
                                
                                # Lineare Beschleunigung
                                elif osc_address == '/linear' and len(message) == 5:
                                    current_linacc_data = {
                                        'lin_acc_x': message[2],
                                        'lin_acc_y': message[3], 
                                        'lin_acc_z': message[4],
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
    stroke_types = {
        '1': 'vorhand_topspin',
        '2': 'vorhand_schupf',
        '3': 'vorhand_block',
        '4': 'rueckhand_topspin',
        '5': 'rueckhand_schupf',
        '6': 'rueckhand_block'
    }
    
    base_folder = "../../rohdaten"
    
    while True:
        print("\n=== XIO NGIMU Tischtennisschlag Aufnahme ===")
        print("1: Vorhand Topspin")
        print("2: Vorhand Schupf")
        print("3: Vorhand Block")
        print("4: Rückhand Topspin")
        print("5: Rückhand Schupf")
        print("6: Rückhand Block")
        print("q: Beenden")
        
        choice = input("\nWählen Sie eine Option: ")
        
        if choice == 'q':
            break
        elif choice in stroke_types:
            stroke_type = stroke_types[choice]
            output_folder = f"{base_folder}/{stroke_type}"
            
            # Ordner erstellen falls nicht vorhanden
            # os.makedirs(output_folder, exist_ok=True)
            
            # Aufnahmedauer festlegen
            duration = input("\nAufnahmedauer in Sekunden (Standard: 5): ")
            duration = int(duration) if duration else RECORDING_DURATION
            
            # Aufnahme starten
            record_data(duration, output_folder, stroke_type)
            
            # Nochmal weiter aufnehmen?
            while True:
                again = input("\nNoch eine Aufnahme vom gleichen Typ? (j/n): ")
                if again.lower() == 'n':
                    break
                record_data(duration, output_folder, stroke_type)
    
    
