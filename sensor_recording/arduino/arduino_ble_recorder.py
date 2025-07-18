"""
Arduino Nicla Sense ME - Bluetooth zu CSV Export (Windows-kompatibel)
Alternative Version ohne pybluez - verwendet Windows Bluetooth API

Voraussetzungen:
- bleak (moderne Bluetooth-Bibliothek): pip install bleak
- asyncio (bereits in Python enthalten)

Verwendung:
1. Arduino Nicla mit Bluetooth-Code geladen
2. Dieses Skript ausführen
3. Gerät wird automatisch erkannt und verbunden
"""

import asyncio
import csv
import time
from datetime import datetime
import sys
import os

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    print("ERROR: bleak nicht installiert!")
    print("\nInstallation:")
    print("pip install bleak")
    sys.exit(1)

# Konstanten - identisch zum USB-Skript
RECORDING_DURATION = 5  # Sekunden
STARTUP_DELAY = 3  # Sekunden
HEADER = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Mag_X', 'Mag_Y', 'Mag_Z', 'Pressure', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']
DATA_LENGTH = 15

# Nordic UART Service UUIDs (Kleinbuchstaben für Kompatibilität)
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Arduino sendet hier
UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Arduino empfängt hier

class BluetoothCSVLogger:
    def __init__(self):
        self.client = None
        self.csv_writer = None
        self.csvfile = None
        self.data_buffer = ""
        self.lines_written = 0
        self.invalid_lines = 0
        self.start_time = None
        self.filename = None
        
    async def scan_for_nicla(self):
        """Suche nach Arduino Nicla Sense ME Geräten"""
        print("Scanne nach Arduino Nicla Sense ME...")
        print("Dies kann 10-15 Sekunden dauern...")
        
        devices = await BleakScanner.discover(timeout=15.0)
        
        nicla_devices = []
        
        print(f"\nGefundene BLE-Geräte ({len(devices)}):")
        for device in devices:
            device_name = device.name or "Unbekannt"
            print(f"  {device_name} ({device.address})")
            
            # Suche nach Nicla-Geräten (case-insensitive)
            if device_name and any(keyword.lower() in device_name.lower() for keyword in 
                                 ["nicla", "arduino", "csv", "sensor"]):
                nicla_devices.append(device)
        
        return nicla_devices
    
    async def connect_to_device(self, device):
        """Verbinde zu einem BLE-Gerät"""
        try:
            print(f"\nVerbinde zu {device.name} ({device.address})...")
            
            # Mehrere Verbindungsversuche mit längerem Timeout
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    self.client = BleakClient(device.address, timeout=20.0)
                    await self.client.connect()
                    
                    if self.client.is_connected:
                        break
                    else:
                        print(f"  Versuch {attempt + 1}/{max_attempts} fehlgeschlagen...")
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(2)
                except Exception as e:
                    print(f"  Verbindungsversuch {attempt + 1} Fehler: {e}")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(2)
            
            if self.client and self.client.is_connected:
                print("✓ Bluetooth-Verbindung erfolgreich!")
                
                # Services anzeigen (Debug)
                try:
                    services = self.client.services
                    if services:
                        service_count = sum(1 for _ in services)
                        print(f"Verfügbare Services: {service_count}")
                        
                        # Debug: Zeige die Services
                        for service in services:
                            print(f"  Service: {service.uuid}")
                            
                        # Prüfe ob UART Service vorhanden ist (case-insensitive)
                        if any(service.uuid.lower() == UART_SERVICE_UUID.lower() for service in services):
                            print(f"✓ UART Service gefunden!")
                        else:
                            print(f"⚠ UART Service ({UART_SERVICE_UUID}) nicht gefunden!")
                except Exception as e:
                    print(f"Warnung beim Service-Check: {e}")
                
                return True
            else:
                print("✗ Verbindung fehlgeschlagen")
                return False
                
        except Exception as e:
            print(f"✗ Verbindungsfehler: {e}")
            return False
    
    def notification_handler(self, sender, data):
        """Handle incoming data from Arduino"""
        try:
            # Bytes zu String konvertieren
            text = data.decode('utf-8')
            self.data_buffer += text
            
            # Vollständige Zeilen verarbeiten
            while '\n' in self.data_buffer:
                line, self.data_buffer = self.data_buffer.split('\n', 1)
                line = line.strip()
                
                if line:
                    self.process_line(line)
                    
        except UnicodeDecodeError as e:
            print(f"Dekodierungsfehler: {e}")
        except Exception as e:
            print(f"Fehler bei Datenverarbeitung: {e}")
    
    def process_line(self, line):
        """Verarbeite eine empfangene Datenzeile"""
        try:
            # Debug-Ausgabe (nur erste 10 Zeilen)
            if self.lines_written < 10:
                print(f"Empfangen: {line}")
            
            # Header-Zeile überspringen
            if line.startswith("Timestamp") or line.startswith("Time"):
                print("Header empfangen - Datenaufzeichnung startet!")
                return
            
            # Leere Zeilen überspringen
            if not line.strip():
                return
            
            # Daten parsen
            data = line.split(',')
            
            if len(data) == DATA_LENGTH:
                # Validierung: Erste Spalte sollte numerisch sein (Timestamp)
                float(data[0])  # Test ob Timestamp numerisch
                
                if self.csv_writer:
                    self.csv_writer.writerow(data)
                    self.lines_written += 1
                    
                    # Status alle 100 Zeilen
                    if self.lines_written % 100 == 0:
                        elapsed = time.time() - self.start_time
                        remaining = RECORDING_DURATION - elapsed
                        print(f"Zeilen: {self.lines_written}, Zeit: {remaining:.1f}s verbleibend")
                    
                    # Buffer leeren
                    if self.csvfile:
                        self.csvfile.flush()
            else:
                self.invalid_lines += 1
                if self.invalid_lines < 5:
                    print(f"Ungültige Zeile (Länge {len(data)}): {line}")
        
        except (ValueError, IndexError):
            self.invalid_lines += 1
            if self.invalid_lines < 5:
                print(f"Parse-Fehler: {line}")
    
    async def start_logging(self):
        """Starte die Datenaufzeichnung"""
        try:
            # Notifications aktivieren
            await self.client.start_notify(UART_TX_CHAR_UUID, self.notification_handler)
            print("✓ Notifications aktiviert - bereit für Daten!")
            
            # CSV-Datei vorbereiten
            os.makedirs('data', exist_ok=True)
            self.filename = f"data/sensordaten_ble_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            self.csvfile = open(self.filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csvfile)
            self.csv_writer.writerow(HEADER)
            
            print(f"\nStarte {RECORDING_DURATION}-Sekunden-Aufzeichnung...")
            print("Drücken Sie Ctrl+C zum vorzeitigen Beenden...")
            
            self.start_time = time.time()
            
            # Warte auf Daten
            await asyncio.sleep(STARTUP_DELAY)
            
            # Aufzeichnung für RECORDING_DURATION
            while time.time() - self.start_time < RECORDING_DURATION:
                await asyncio.sleep(0.1)  # 100ms Pause
                
                # Verbindung prüfen
                if not self.client.is_connected:
                    print("⚠ Verbindung verloren!")
                    break
            
        except KeyboardInterrupt:
            print(f"\n\nAufzeichnung durch Benutzer beendet!")
        except Exception as e:
            print(f"\nFehler während der Aufzeichnung: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Aufräumen und Verbindung schließen"""
        try:
            if self.client and self.client.is_connected:
                await self.client.stop_notify(UART_TX_CHAR_UUID)
                await self.client.disconnect()
                print("✓ Bluetooth-Verbindung geschlossen")
        except:
            pass
        
        if self.csvfile:
            self.csvfile.close()
        
        print(f"\n" + "=" * 50)
        print("AUFZEICHNUNG BEENDET")
        print(f"✓ {self.lines_written} gültige Datenzeilen geschrieben")
        if self.invalid_lines > 0:
            print(f"⚠ {self.invalid_lines} ungültige Zeilen übersprungen")
        if self.filename:
            print(f"✓ Datei gespeichert: {self.filename}")
        print(f"=" * 50)
        
        if self.lines_written > 0:
            print(f"\nZum Visualisieren:")
            print(f"python visualize_data.py {self.filename}")
            print(f"\nZum Konvertieren der Einheiten:")
            print(f"python convert_units.py {self.filename}")
        else:
            print("\n⚠ Keine Daten empfangen!")
            print("Tipps:")
            print("- Läuft der richtige Code auf dem Arduino?")
            print("- Sendet das Arduino Daten über Bluetooth?")
            print("- Ist die Bluetooth-Verbindung stabil?")

async def main():
    print("=" * 50)
    print("Arduino Nicla Sense ME - Bluetooth CSV Export")
    print("Moderne Version mit bleak (ohne pybluez)")
    print("=" * 50)
    
    logger = BluetoothCSVLogger()
    
    try:
        # Nach Geräten suchen
        nicla_devices = await logger.scan_for_nicla()
        
        if not nicla_devices:
            print("\n⚠ Keine Arduino Nicla Sense ME Geräte gefunden!")
            print("\nTipps:")
            print("- Ist das Arduino Nicla eingeschaltet?")
            print("- Ist der Bluetooth-Code geladen?")
            print("- Läuft das Arduino und blinkt grün?")
            return
        
        # Gerät auswählen
        if len(nicla_devices) == 1:
            selected_device = nicla_devices[0]
            print(f"\n✓ Automatisch ausgewählt: {selected_device.name}")
        else:
            print(f"\nMehrere Nicla-Geräte gefunden:")
            for i, device in enumerate(nicla_devices):
                print(f"{i+1}. {device.name} ({device.address})")
            
            try:
                choice = int(input(f"Wählen Sie ein Gerät (1-{len(nicla_devices)}): ")) - 1
                if 0 <= choice < len(nicla_devices):
                    selected_device = nicla_devices[choice]
                else:
                    print("Ungültige Auswahl!")
                    return
            except ValueError:
                print("Ungültige Eingabe!")
                return
        
        # Verbinden und loggen
        if await logger.connect_to_device(selected_device):
            await logger.start_logging()
    
    except KeyboardInterrupt:
        print("\n\nProgramm durch Benutzer beendet!")
    except Exception as e:
        print(f"\nUnerwarteter Fehler: {e}")
    finally:
        await logger.cleanup()

if __name__ == "__main__":
    # asyncio für Windows konfigurieren
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
