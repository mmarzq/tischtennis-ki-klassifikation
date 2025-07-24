"""
Arduino Nicla Bluetooth to CSV Logger für Tischtennisschlag-Klassifikation
Receives sensor data over Bluetooth and saves to CSV file

Requirements: pip install bleak
"""

import asyncio
import csv
import time
from datetime import datetime
import os
from bleak import BleakScanner, BleakClient

# Konstanten
RECORDING_DURATION = 5  # Sekunden
UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# Global BLE (Bluetooth Low Energy) client
ble_client = None

# CSV Header (gleich wie bei NGIMU)
HEADER = ['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Mag_X', 'Mag_Y', 'Mag_Z',
           'Bar', 'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z', 'Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z']

# Global variables
csv_writer = None
csv_file = None
line_count = 0
start_time = None
first_timestamp = None          # for relative timestamp calculation 

def handle_data(sender, data, recording_duration):
    """Process incoming BLE data and write to CSV"""
    global csv_writer, line_count, first_timestamp
    
    # Convert bytes to text
    text = data.decode('utf-8').strip()
    
    # Split data and write to CSV
    values = text.split(',')
    if len(values) == 18:  # Expected number of values
        """first method: relative to first_timestamp (time from Arduino) """
        # Convert timestamp to start from 0
        original_timestamp = int(values[0])
        if first_timestamp is None:
            first_timestamp = original_timestamp
        
        relative_timestamp = (original_timestamp - first_timestamp) / 1000.0
        values[0] = relative_timestamp
        
        csv_writer.writerow(values)
        line_count += 1

async def connect_to_nicla():
    """Connect to Arduino Nicla device once"""
    global ble_client
    
    if ble_client and ble_client.is_connected:
        return True
    
    print("Scanning for Arduino Nicla...")
    devices = await BleakScanner.discover(timeout=10.0)
    
    # Look for Nicla device
    for device in devices:
        if device.name and "nicla" in device.name.lower():
            print(f"Found: {device.name} ({device.address})")
            ble_client = BleakClient(device.address)
            try:
                await ble_client.connect()
                print("Connected!")
                return True
            except Exception as e:
                print(f"Connection failed: {e}")
                return False
    
    print("No Nicla device found!")
    return False

async def disconnect_nicla():
    """Disconnect from Nicla device"""
    global ble_client
    if ble_client and ble_client.is_connected:
        await ble_client.disconnect()
        print("Disconnected from device")

async def record_data(duration, output_folder, stroke_type):
    """Record sensor data for specified duration and stroke type"""
    global csv_writer, csv_file, start_time, first_timestamp, line_count, ble_client
    
    # Reset for new recording
    first_timestamp = None
    line_count = 0
    
    # Ensure connection exists
    if not ble_client or not ble_client.is_connected:
        if not await connect_to_nicla():
            return None
    
    try:
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_folder}/{stroke_type}_{timestamp}_arduino.csv"
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(HEADER)
        
        # Hinweis für die Benutzer
        print(f"Schlagtyp: {stroke_type}")
        print(f"Starte Aufzeichnung für {duration} Sekunden...")
        print("2 ...")
        await asyncio.sleep(1)  
        print("1 ...")
        await asyncio.sleep(1)
        print("0 ..")
        
        # Start receiving data with duration parameter
        handler = lambda sender, data: handle_data(sender, data, duration)
        await ble_client.start_notify(UART_TX_CHAR, handler)
        
        # Record for specified time
        start_time = time.time()
        await asyncio.sleep(duration)
        
        # Stop notifications (but keep connection)
        await ble_client.stop_notify(UART_TX_CHAR)
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    finally:
        if csv_file:
            csv_file.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nAufzeichnung beendet!")
    print(f"Dauer: {elapsed_time:.1f} Sekunden")
    print(f"Datensätze: {line_count}")
    if line_count > 0:
        print(f"Rate: {line_count/elapsed_time:.1f} Hz")
    
    return filename


async def main():
    """
    Hauptmenü für die Schlagtypauswahl
    """
    stroke_types = {
        '1': 'vorhand_topspin',
        '2': 'vorhand_schupf',
        '3': 'rueckhand_topspin',
        '4': 'rueckhand_schupf'
    }
    
    base_folder = "../../rohdaten"
    
    try:
        while True:
            print("\n=== Arduino Nicla Tischtennisschlag Aufnahme ===")
            print("1: Vorhand Topspin")
            print("2: Vorhand Schupf")
            print("3: Rückhand Topspin")
            print("4: Rückhand Schupf")
            print("q: Beenden")
            
            choice = input("\nWählen Sie eine Option: ")
            
            if choice == 'q':
                break
            elif choice in stroke_types:
                stroke_type = stroke_types[choice]
                output_folder = f"{base_folder}/{stroke_type}"
                
                # Aufnahmedauer festlegen
                duration_input = input("\nAufnahmedauer in Sekunden (Standard: 5): ")
                duration = int(duration_input) if duration_input else RECORDING_DURATION
                
                # Aufnahme starten
                await record_data(duration, output_folder, stroke_type)
                
                # Nochmal weiter aufnehmen?
                while True:
                    again = input("\nNoch eine Aufnahme vom gleichen Typ? (j/n): ")
                    if again.lower() == 'n':
                        break
                    await record_data(duration, output_folder, stroke_type)
    
    finally:
        # Disconnect when exiting
        await disconnect_nicla()

if __name__ == "__main__":
    asyncio.run(main())