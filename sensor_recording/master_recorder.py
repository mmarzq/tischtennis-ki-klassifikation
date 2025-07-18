"""
Master-Aufnahmeskript für beide Sensoren
Ermöglicht die Auswahl zwischen Arduino und NGIMU
"""

import os
import sys
import subprocess

def print_banner():
    """Zeigt das Projekt-Banner"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     Tischtennisschlag-Klassifikation mit KI          ║
    ║              Sensor-Datenerfassung                    ║
    ╚═══════════════════════════════════════════════════════╝
    """)

def check_sensor_connection():
    """Überprüft verfügbare Sensoren"""
    print("\n🔍 Suche nach verfügbaren Sensoren...")
    
    available_sensors = []
    
    # Arduino Check (über COM-Ports)
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if "Arduino" in port.description or "Nicla" in port.description:
                available_sensors.append(('arduino', port.device))
                print(f"✓ Arduino Nicla Sense ME gefunden auf {port.device}")
    except:
        pass
    
    # NGIMU Check (Netzwerk)
    print("✓ X-IO NGIMU (Netzwerk) - Manuelle Konfiguration erforderlich")
    available_sensors.append(('ngimu', 'network'))
    
    return available_sensors

def arduino_recording():
    """Startet Arduino-Aufnahme"""
    print("\n📱 Arduino Nicla Sense ME Aufnahme")
    print("="*40)
    
    # Prüfen ob Sketch geladen ist
    print("\n⚠️  Stellen Sie sicher, dass:")
    print("1. Der Arduino-Sketch 'nicla_sense_me_recorder.ino' geladen ist")
    print("2. Der richtige COM-Port ausgewählt ist")
    print("3. Die Baudrate auf 115200 eingestellt ist")
    
    input("\nDrücken Sie Enter wenn bereit...")
    
    # Arduino Recorder starten
    os.chdir('sensor_recording')
    subprocess.run([sys.executable, 'arduino_recorder.py'])
    os.chdir('..')

def ngimu_recording():
    """Startet NGIMU-Aufnahme"""
    print("\n🌐 X-IO NGIMU Aufnahme")
    print("="*40)
    
    print("\n⚠️  Stellen Sie sicher, dass:")
    print("1. NGIMU eingeschaltet und im Netzwerk ist")
    print("2. OSC-Ausgabe in NGIMU GUI aktiviert ist")
    print("3. Send Rates konfiguriert sind (Sensors: 100Hz)")
    print("4. Die IP-Adresse bekannt ist")
    
    # Optionen anbieten
    print("\nWas möchten Sie tun?")
    print("1: Direkt zur Datenaufnahme (OSC)")
    print("2: Sensor konfigurieren")
    print("3: Live-Visualisierung testen")
    print("4: CSV-Dateien konvertieren")
    
    choice = input("\nWahl: ")
    
    os.chdir('sensor_recording/xio_ngimu')
    
    if choice == '1':
        subprocess.run([sys.executable, 'ngimu_osc_recorder.py'])
    elif choice == '2':
        subprocess.run([sys.executable, 'ngimu_configurator.py'])
    elif choice == '3':
        subprocess.run([sys.executable, 'ngimu_realtime_visualizer.py'])
    elif choice == '4':
        subprocess.run([sys.executable, 'ngimu_csv_converter.py'])
    
    os.chdir('../..')

def sensor_comparison():
    """Startet Sensor-Vergleich"""
    print("\n📊 Sensor-Vergleichstool")
    print("="*40)
    
    os.chdir('sensor_recording')
    subprocess.run([sys.executable, 'sensor_comparison.py'])
    os.chdir('..')

def main():
    """Hauptmenü"""
    print_banner()
    
    # Sensoren suchen
    available_sensors = check_sensor_connection()
    
    while True:
        print("\n🎯 HAUPTMENÜ")
        print("="*40)
        print("1: Arduino Nicla Sense ME - Datenaufnahme")
        print("2: X-IO NGIMU - Datenaufnahme & Konfiguration")
        print("3: Sensor-Vergleich (Arduino vs NGIMU)")
        print("4: Daten vorverarbeiten")
        print("5: Daten visualisieren")
        print("q: Beenden")
        
        choice = input("\nWählen Sie eine Option: ")
        
        if choice == 'q':
            print("\nAuf Wiedersehen! 🏓")
            break
        elif choice == '1':
            arduino_recording()
        elif choice == '2':
            ngimu_recording()
        elif choice == '3':
            sensor_comparison()
        elif choice == '4':
            print("\nStarte Datenvorverarbeitung...")
            subprocess.run([sys.executable, 'data_preprocessing.py'])
        elif choice == '5':
            print("\nStarte Visualisierung...")
            subprocess.run([sys.executable, 'visualize_data.py'])
        else:
            print("Ungültige Eingabe!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgramm beendet.")
