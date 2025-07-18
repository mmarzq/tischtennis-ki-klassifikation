"""
Erweiterter BLE Scanner - Findet alle Bluetooth-Geräte
"""

import asyncio
from bleak import BleakScanner
import sys

async def scan_all():
    print("Scanne nach ALLEN BLE-Geräten...")
    print("Dies dauert 30 Sekunden...\n")
    
    # Mehrere Scans für bessere Erkennung
    all_devices = {}
    
    for i in range(3):
        print(f"Scan {i+1}/3...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.address not in all_devices:
                all_devices[device.address] = device
                name = device.name or "Unbekannt"
                rssi = device.rssi if hasattr(device, 'rssi') else 'N/A'
                print(f"NEU: {name} ({device.address}) RSSI: {rssi}")
    
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"Gefundene Geräte: {len(all_devices)}")
    
    # Suche nach Arduino
    arduino_found = False
    for addr, device in all_devices.items():
        name = device.name or ""
        if any(keyword in name.lower() for keyword in ['nicla', 'arduino', 'test', 'csv']):
            print(f"\n✓ MÖGLICHES ARDUINO: {name} ({addr})")
            arduino_found = True
    
    if not arduino_found:
        print("\n⚠ Kein Arduino gefunden!")
        print("\nMögliche MAC-Adressen (ohne Namen):")
        for addr, device in all_devices.items():
            if not device.name:
                print(f"  {addr}")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(scan_all())
