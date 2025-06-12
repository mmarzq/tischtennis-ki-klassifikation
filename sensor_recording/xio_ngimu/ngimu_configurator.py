"""
NGIMU Konfigurations-Tool
Konfiguriert NGIMU-Einstellungen über OSC-Befehle
"""

from pythonosc import udp_client
import time

class NGIMUConfigurator:
    def __init__(self, ip, port=9000):
        self.ip = ip
        self.port = port
        self.client = udp_client.SimpleUDPClient(ip, port)
        
    def set_send_rate(self, data_type, rate):
        """Setzt die Senderate für einen Datentyp"""
        command = f"/rate/{data_type}"
        self.client.send_message(command, rate)
        print(f"Setze {data_type} Rate auf {rate} Hz")
        time.sleep(0.1)
    
    def configure_for_tabletennis(self):
        """Optimale Konfiguration für Tischtennis"""
        print("\nKonfiguriere NGIMU für Tischtennis-Analyse...")
        
        # Sensoren mit hoher Rate für schnelle Bewegungen
        self.set_send_rate("sensors", 100.0)
        
        # Quaternion für Orientierung
        self.set_send_rate("quaternion", 100.0)
        
        # Lineare Beschleunigung (optional aber hilfreich)
        self.set_send_rate("linear", 100.0)
        
        # Euler nur für Debug
        self.set_send_rate("euler", 50.0)
        
        # Andere deaktivieren für bessere Performance
        self.set_send_rate("earth", 0.0)
        self.set_send_rate("altitude", 0.0)
        self.set_send_rate("temperature", 0.0)
        self.set_send_rate("humidity", 0.0)
        self.set_send_rate("magnitudes", 0.0)
        
        print("\nKonfiguration abgeschlossen!")
        print("Optimiert für:")
        print("- Sensors: 100 Hz (Gyro, Acc, Mag)")
        print("- Quaternion: 100 Hz (Orientierung)")
        print("- Linear: 100 Hz (Lineare Beschleunigung)")
        print("- Euler: 50 Hz (Debug)")
    
    def set_wifi_settings(self, mode="client", ssid=None, password=None):
        """WiFi-Einstellungen konfigurieren"""
        if mode == "ap":
            # Access Point Modus
            self.client.send_message("/wifi/mode", "ap")
            print("WiFi-Modus: Access Point")
        else:
            # Client Modus
            self.client.send_message("/wifi/mode", "client")
            if ssid:
                self.client.send_message("/wifi/client/ssid", ssid)
                if password:
                    self.client.send_message("/wifi/client/key", password)
                print(f"WiFi-Client: Verbinde mit {ssid}")
    
    def calibrate_gyroscope(self):
        """Gyroskop kalibrieren"""
        print("\nGyroskop-Kalibrierung...")
        print("Bitte NGIMU ruhig halten!")
        time.sleep(2)
        
        self.client.send_message("/calibrate/gyroscope", 1)
        print("Kalibrierung gestartet...")
        time.sleep(3)
        print("Kalibrierung abgeschlossen!")
    
    def reset_heading(self):
        """Heading/Ausrichtung zurücksetzen"""
        self.client.send_message("/reset/heading", 1)
        print("Heading zurückgesetzt")
    
    def get_battery_status(self):
        """Batteriestatus abfragen"""
        self.client.send_message("/battery", 1)
        print("Batteriestatus angefordert")
    
    def interactive_menu(self):
        """Interaktives Konfigurationsmenü"""
        while True:
            print("\n=== NGIMU Konfiguration ===")
            print("1: Tischtennis-Optimierung anwenden")
            print("2: Gyroskop kalibrieren")
            print("3: Heading zurücksetzen")
            print("4: Batteriestatus")
            print("5: Manuelle Send-Rate einstellen")
            print("q: Beenden")
            
            choice = input("\nWählen Sie eine Option: ")
            
            if choice == 'q':
                break
            elif choice == '1':
                self.configure_for_tabletennis()
            elif choice == '2':
                self.calibrate_gyroscope()
            elif choice == '3':
                self.reset_heading()
            elif choice == '4':
                self.get_battery_status()
            elif choice == '5':
                data_type = input("Datentyp (sensors/quaternion/euler/linear): ")
                rate = float(input("Rate in Hz (0 zum Deaktivieren): "))
                self.set_send_rate(data_type, rate)

def main():
    """Hauptfunktion"""
    print("=== NGIMU Konfigurations-Tool ===")
    
    # IP-Adresse abfragen
    ip = input("NGIMU IP-Adresse (z.B. 192.168.1.100): ")
    if not ip:
        print("Verwende Standard: 192.168.1.1")
        ip = "192.168.1.1"
    
    # Konfigurator erstellen
    config = NGIMUConfigurator(ip)
    
    # Menü starten
    config.interactive_menu()
    
    print("\nKonfiguration beendet.")

if __name__ == "__main__":
    main()
