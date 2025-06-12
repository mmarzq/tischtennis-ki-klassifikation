"""
NGIMU Debug und Verbindungstest
Hilft bei der Diagnose von Verbindungsproblemen
"""

import socket
import time
from pythonosc import dispatcher, osc_server
import threading
import subprocess
import platform

class NGIMUDebugger:
    def __init__(self):
        self.data_received = False
        self.server = None
        
    def get_local_ip(self):
        """Ermittelt die lokale IP-Adresse"""
        try:
            # Verbindung zu externer IP um lokale IP zu finden
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def check_network_info(self):
        """Zeigt Netzwerkinformationen"""
        print("\n📡 NETZWERK-INFORMATIONEN:")
        print("="*50)
        
        # Lokale IP
        local_ip = self.get_local_ip()
        print(f"Ihre IP-Adresse: {local_ip}")
        
        # Hostname
        hostname = socket.gethostname()
        print(f"Computer-Name: {hostname}")
        
        # Alle IPs
        print("\nAlle Netzwerk-Interfaces:")
        try:
            for ip in socket.gethostbyname_ex(hostname)[2]:
                print(f"  - {ip}")
        except:
            print("  Fehler beim Abrufen der Interfaces")
        
        return local_ip
    
    def ping_device(self, ip):
        """Pingt ein Gerät"""
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", ip]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def scan_for_ngimu(self):
        """Scannt das Netzwerk nach NGIMU"""
        print("\n🔍 SUCHE NGIMU IM NETZWERK...")
        print("="*50)
        
        local_ip = self.get_local_ip()
        network_prefix = '.'.join(local_ip.split('.')[:-1])
        
        # Typische NGIMU IPs
        common_ips = [
            "192.168.1.1",      # NGIMU Default AP Mode
            f"{network_prefix}.1",
            f"{network_prefix}.100",
            f"{network_prefix}.101",
        ]
        
        print("Teste bekannte NGIMU IP-Adressen:")
        found_devices = []
        
        for ip in common_ips:
            print(f"  Teste {ip}...", end="", flush=True)
            if self.ping_device(ip):
                print(" ✓ Gerät gefunden!")
                found_devices.append(ip)
            else:
                print(" ✗")
        
        if found_devices:
            print(f"\n✓ Mögliche NGIMU-Geräte gefunden: {found_devices}")
        else:
            print("\n✗ Keine NGIMU im Netzwerk gefunden")
            print("\nMögliche Lösungen:")
            print("1. NGIMU im AP-Modus? Verbinden Sie sich mit 'NGIMU' WiFi")
            print("2. Firewall blockiert? Deaktivieren Sie temporär die Firewall")
            print("3. Falsches Netzwerk? Prüfen Sie die WiFi-Verbindung")
        
        return found_devices
    
    def test_osc_ports(self):
        """Testet verschiedene OSC-Ports"""
        print("\n🔌 TESTE OSC-PORTS...")
        print("="*50)
        
        test_ports = [9000, 8000, 7000, 1234]
        open_ports = []
        
        for port in test_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(("0.0.0.0", port))
                sock.close()
                print(f"  Port {port}: ✓ Verfügbar")
                open_ports.append(port)
            except:
                print(f"  Port {port}: ✗ Belegt")
        
        return open_ports
    
    def handle_any_osc(self, address, *args):
        """Empfängt alle OSC-Nachrichten"""
        self.data_received = True
        print(f"\n📨 OSC empfangen: {address}")
        print(f"   Daten: {args[:5]}...")  # Erste 5 Werte
    
    def listen_for_osc(self, port=9000, duration=10):
        """Lauscht auf OSC-Nachrichten"""
        print(f"\n👂 LAUSCHE AUF OSC-NACHRICHTEN (Port {port})...")
        print(f"Warte {duration} Sekunden auf Daten...")
        print("="*50)
        
        # Dispatcher für alle Nachrichten
        disp = dispatcher.Dispatcher()
        disp.set_default_handler(self.handle_any_osc)
        
        # Server starten
        server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", port), disp)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Warten
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.data_received:
                print("\n✓ NGIMU sendet Daten!")
                break
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        server.shutdown()
        
        if not self.data_received:
            print("\n\n✗ Keine OSC-Daten empfangen")
        
        return self.data_received
    
    def show_ngimu_settings(self):
        """Zeigt empfohlene NGIMU-Einstellungen"""
        print("\n⚙️  EMPFOHLENE NGIMU-EINSTELLUNGEN:")
        print("="*50)
        print("In NGIMU GUI:")
        print("\n1. Settings → WiFi:")
        print("   - Mode: Client (für Ihr Netzwerk)")
        print("   - ODER Mode: AP (eigenes Netzwerk)")
        print("\n2. Settings → Send Rates:")
        print("   - Sensors: 100")
        print("   - Quaternion: 100")
        print("   - Linear: 100")
        print("   - Andere: 0")
        print("\n3. Settings → Network Announce:")
        print("   - Destination IP: Broadcast (255.255.255.255)")
        print(f"   - ODER spezifisch: {self.get_local_ip()}")
        print("   - Port: 9000")
        print("\n4. Klicken Sie 'Send' für jede Einstellung!")
    
    def interactive_debug(self):
        """Interaktives Debug-Menü"""
        while True:
            print("\n🔧 NGIMU DEBUG-MENÜ")
            print("="*50)
            print("1: Netzwerk-Info anzeigen")
            print("2: Nach NGIMU suchen")
            print("3: OSC-Ports testen")
            print("4: Auf OSC-Daten lauschen")
            print("5: NGIMU-Einstellungen anzeigen")
            print("6: Kompletter Systemtest")
            print("q: Beenden")
            
            choice = input("\nWahl: ")
            
            if choice == 'q':
                break
            elif choice == '1':
                self.check_network_info()
            elif choice == '2':
                self.scan_for_ngimu()
            elif choice == '3':
                self.test_osc_ports()
            elif choice == '4':
                port = input("Port (Standard 9000): ")
                port = int(port) if port else 9000
                self.listen_for_osc(port)
            elif choice == '5':
                self.show_ngimu_settings()
            elif choice == '6':
                self.full_system_test()
    
    def full_system_test(self):
        """Führt einen kompletten Systemtest durch"""
        print("\n🚀 KOMPLETTER SYSTEMTEST")
        print("="*50)
        
        # 1. Netzwerk
        local_ip = self.check_network_info()
        
        # 2. Ports
        print("\n")
        open_ports = self.test_osc_ports()
        
        # 3. NGIMU suchen
        print("\n")
        devices = self.scan_for_ngimu()
        
        # 4. OSC lauschen
        if open_ports:
            print("\n")
            self.listen_for_osc(open_ports[0])
        
        # 5. Zusammenfassung
        print("\n📊 ZUSAMMENFASSUNG:")
        print("="*50)
        print(f"✓ Ihre IP: {local_ip}")
        print(f"✓ Verfügbare Ports: {open_ports}")
        
        if devices:
            print(f"✓ Mögliche NGIMUs: {devices}")
            print("\n➡️  Nächste Schritte:")
            print("1. Öffnen Sie NGIMU GUI")
            print(f"2. Setzen Sie Destination IP auf: {local_ip}")
            print("3. Setzen Sie Port auf: 9000")
            print("4. Aktivieren Sie Send Rates")
        else:
            print("✗ Kein NGIMU gefunden")
            print("\n➡️  Nächste Schritte:")
            print("1. Prüfen Sie die NGIMU WiFi-Verbindung")
            print("2. Versuchen Sie NGIMU im AP-Modus")
            print("3. Deaktivieren Sie die Firewall temporär")

def main():
    """Hauptfunktion"""
    print("=== NGIMU DEBUG & VERBINDUNGSTEST ===")
    print("Dieses Tool hilft bei Verbindungsproblemen\n")
    
    debugger = NGIMUDebugger()
    
    # Quick-Test oder Menü?
    print("Was möchten Sie tun?")
    print("1: Schnelltest (empfohlen)")
    print("2: Debug-Menü")
    
    choice = input("\nWahl: ")
    
    if choice == '1':
        debugger.full_system_test()
    else:
        debugger.interactive_debug()

if __name__ == "__main__":
    main()
