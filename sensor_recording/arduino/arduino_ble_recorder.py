"""
Simplified Arduino Nicla Bluetooth to CSV Logger
Receives sensor data over Bluetooth and saves to CSV file

Requirements: pip install bleak
"""

import asyncio
import csv
import time
from datetime import datetime
import os
from bleak import BleakScanner, BleakClient

# Settings
RECORDING_TIME = 5  # seconds
UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# Global variables
csv_writer = None
csv_file = None
line_count = 0
start_time = None
first_timestamp = None          # for relative timestamp calculation => 

def handle_data(sender, data):
    """Process incoming Bluetooth data"""
    global csv_writer, line_count, first_timestamp
    
    # Convert bytes to text
    text = data.decode('utf-8').strip()
    
    # Skip empty lines or header
    if not text or text.startswith("Timestamp"):
        return
    
    # Split data and write to CSV
    values = text.split(',')
    if len(values) == 18:  # Expected number of values
        """first method: relative to first_timestamp (time from Arduino) """
        # Convert timestamp to start from 0
        original_timestamp = int(values[0])
        
        # Set first timestamp on first data
        if first_timestamp is None:
            first_timestamp = original_timestamp
        
        # Calculate relative timestamp in seconds
        relative_timestamp = (original_timestamp - first_timestamp) / 1000.0
        values[0] = relative_timestamp   #f"{relative_timestamp:.5f}"

        """second method: timestamp from Python side, first_timestamp is not needed"""
        """
        if start_time is None:
            start_time = time.time()
        current_timestamp = time.time() - start_time
        values[0] = f"{current_timestamp:.3f}"
        """

        csv_writer.writerow(values)
        line_count += 1

        # Show progress
        if line_count % 100 == 0:
            time_left = RECORDING_TIME - (time.time() - start_time)
            print(f"Lines: {line_count}, Time left: {time_left:.1f}s")

async def find_nicla():
    """Find Arduino Nicla device"""
    print("Scanning for Arduino Nicla...")
    devices = await BleakScanner.discover(timeout=10.0)
    
    # Look for Nicla device
    for device in devices:
        if device.name and "nicla" in device.name.lower():
            print(f"Found: {device.name} ({device.address})")
            return device
    
    print("No Nicla device found!")
    return None

async def main():
    global csv_writer, csv_file, start_time, first_timestamp
    
    print("Arduino Nicla Bluetooth CSV Logger")
    print("-" * 30)
    
    # Reset first timestamp for each recording
    first_timestamp = None
    
    # Find device
    device = await find_nicla()
    if not device:
        return
    
    # Connect
    print(f"Connecting to {device.name}...")
    client = BleakClient(device.address)
    
    try:
        await client.connect()
        print("Connected!")
        
        # Create CSV file
        os.makedirs('data', exist_ok=True)
        filename = f"data/sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['Timestamp', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 
                           'Acc_X', 'Acc_Y', 'Acc_Z', 
                           'Mag_X', 'Mag_Y', 'Mag_Z', 
                           'Bar', 
                           'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',
                           'Lin_Acc_X', 'Lin_Acc_Y', 'Lin_Acc_Z'])
        
        # Start receiving data
        await client.start_notify(UART_TX_CHAR, handle_data)
        
        # Record for specified time
        print(f"Recording for {RECORDING_TIME} seconds...")
        start_time = time.time()
        await asyncio.sleep(RECORDING_TIME)
        
        # Stop and disconnect
        await client.stop_notify(UART_TX_CHAR)
        await client.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if csv_file:
            csv_file.close()
            print(f"\nDone! Saved {line_count} lines to {filename}")

if __name__ == "__main__":
    asyncio.run(main())