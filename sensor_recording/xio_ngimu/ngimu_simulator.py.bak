"""
Simulierter NGIMU für Tests ohne echte Hardware
Sendet realistische Tischtennisdaten über OSC
"""

from pythonosc import udp_client
import time
import numpy as np
import threading

class NGIMUSimulator:
    def __init__(self, target_ip="127.0.0.1", target_port=9000):
        self.client = udp_client.SimpleUDPClient(target_ip, target_port)
        self.running = False
        self.stroke_type = "idle"
        
    def generate_sensor_data(self, t):
        """Generiert realistische Sensordaten basierend auf Schlagtyp"""
        
        if self.stroke_type == "idle":
            # Ruhezustand
            gyro = [
                np.random.normal(0, 0.5),
                np.random.normal(0, 0.5),
                np.random.normal(0, 0.5)
            ]
            acc = [
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(1.0, 0.01)  # Gravitation
            ]
            
        elif self.stroke_type == "vorhand_topspin":
            # Vorhand Topspin Muster
            phase = t % 2.0  # 2 Sekunden Zyklus
            
            if phase < 0.5:  # Ausholbewegung
                gyro = [
                    -50 * np.sin(phase * np.pi * 4),
                    20 * np.sin(phase * np.pi * 2),
                    10 * np.sin(phase * np.pi * 2)
                ]
                acc = [
                    0.5 * np.sin(phase * np.pi * 2),
                    0.3 * np.cos(phase * np.pi * 2),
                    1.0 + 0.2 * np.sin(phase * np.pi)
                ]
            elif phase < 1.0:  # Schlag
                gyro = [
                    200 * np.sin((phase - 0.5) * np.pi * 4),
                    100 * np.sin((phase - 0.5) * np.pi * 4),
                    50 * np.sin((phase - 0.5) * np.pi * 2)
                ]
                acc = [
                    2.0 * np.sin((phase - 0.5) * np.pi * 2),
                    1.5 * np.cos((phase - 0.5) * np.pi * 2),
                    1.0 + 1.0 * np.sin((phase - 0.5) * np.pi)
                ]
            else:  # Ausschwung
                gyro = [
                    50 * np.exp(-(phase - 1.0) * 3),
                    20 * np.exp(-(phase - 1.0) * 3),
                    10 * np.exp(-(phase - 1.0) * 3)
                ]
                acc = [
                    0.3 * np.exp(-(phase - 1.0) * 2),
                    0.2 * np.exp(-(phase - 1.0) * 2),
                    1.0
                ]
                
        # Rauschen hinzufügen
        gyro = [g + np.random.normal(0, 1) for g in gyro]
        acc = [a + np.random.normal(0, 0.02) for a in acc]
        
        # Magnetometer (konstant + leichtes Rauschen)
        mag = [
            25.0 + np.random.normal(0, 0.5),
            -12.0 + np.random.normal(0, 0.5),
            45.0 + np.random.normal(0, 0.5)
        ]
        
        return gyro, acc, mag
    
    def send_loop(self):
        """Hauptschleife für Datenübertragung"""
        start_time = time.time()
        sample_period = 0.01  # 100Hz
        
        while self.running:
            current_time = time.time() - start_time
            
            # Sensordaten generieren
            gyro, acc, mag = self.generate_sensor_data(current_time)
            
            # OSC-Nachricht senden
            # Format: timestamp, gyroX, gyroY, gyroZ, accX, accY, accZ, magX, magY, magZ
            self.client.send_message("/sensors", 
                [current_time] + gyro + acc + mag)
            
            # Quaternion senden (vereinfacht)
            self.client.send_message("/quaternion", [1.0, 0.0, 0.0, 0.0])
            
            # Sample-Rate einhalten
            time.sleep(sample_period)
    
    def start(self):
        """Startet die Simulation"""
        self.running = True
        self.thread = threading.Thread(target=self.send_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Simulator gestartet")
    
    def stop(self):
        """Stoppt die Simulation"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        print("Simulator gestoppt")
    
    def interactive_control(self):
        """Interaktive Steuerung der Simulation"""
        print("\n=== NGIMU SIMULATOR ===")
        print(f"Sende an {self.client._address}:{self.client._port}")
        print("\nBefehle:")
        print("1: Ruhezustand")
        print("2: Vorhand Topspin")
        print("3: Zufällige Bewegung")
        print("s: Status")
        print("q: Beenden")
        
        self.start()
        
        try:
            while True:
                cmd = input("\nBefehl: ")
                
                if cmd == 'q':
                    break
                elif cmd == '1':
                    self.stroke_type = "idle"
                    print("→ Ruhezustand")
                elif cmd == '2':
                    self.stroke_type = "vorhand_topspin"
                    print("→ Vorhand Topspin")
                elif cmd == '3':
                    self.stroke_type = "random"
                    print("→ Zufällige Bewegung")
                elif cmd == 's':
                    print(f"Status: Läuft, Modus: {self.stroke_type}")
                
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

def main():
    """Hauptfunktion"""
    import sys
    
    print("=== NGIMU SIMULATOR ===")
    print("Simuliert NGIMU-Sensordaten für Tests\n")
    
    # Ziel-IP und Port
    if len(sys.argv) > 1:
        target_ip = sys.argv[1]
    else:
        target_ip = input("Ziel-IP (Standard: localhost): ") or "127.0.0.1"
    
    if len(sys.argv) > 2:
        target_port = int(sys.argv[2])
    else:
        port_str = input("Ziel-Port (Standard: 9000): ")
        target_port = int(port_str) if port_str else 9000
    
    # Simulator starten
    sim = NGIMUSimulator(target_ip, target_port)
    
    # Modus wählen
    print("\nModus:")
    print("1: Interaktiv (manuell steuern)")
    print("2: Automatisch (kontinuierliche Schläge)")
    
    mode = input("Wahl: ")
    
    if mode == '2':
        print("\nAutomatischer Modus - Sende kontinuierliche Daten")
        print("Drücken Sie Ctrl+C zum Beenden")
        sim.stroke_type = "vorhand_topspin"
        sim.start()
        
        try:
            while True:
                time.sleep(5)
                # Wechsel zwischen Schlagtypen
                sim.stroke_type = "idle" if sim.stroke_type == "vorhand_topspin" else "vorhand_topspin"
                print(f"Wechsel zu: {sim.stroke_type}")
        except KeyboardInterrupt:
            sim.stop()
    else:
        sim.interactive_control()

if __name__ == "__main__":
    main()
