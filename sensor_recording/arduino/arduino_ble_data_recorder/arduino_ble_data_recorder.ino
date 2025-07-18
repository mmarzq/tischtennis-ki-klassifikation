/*
Arduino Nicla Sense ME - Bluetooth Sensor Export
Sends sensor data over Bluetooth using Nordic UART Service
*/

#include "Nicla_System.h"
#include "Arduino_BHY2.h"
#include <ArduinoBLE.h>

// Nordic UART Service
BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLECharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E", BLENotify, 200);

// Sensors
SensorXYZ gyro(SENSOR_ID_GYRO);
SensorXYZ accel(SENSOR_ID_ACC);
SensorXYZ mag(SENSOR_ID_MAG);
Sensor pressure(SENSOR_ID_BARO);
SensorQuaternion quat(SENSOR_ID_RV);
SensorXYZ linAccel(SENSOR_ID_LACC);

bool connected = false;

void setup() {
  Serial.begin(115200);
  
  // Initialize system
  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(red);
  
  // Battery-Charging setup
  nicla::setBatteryNTCEnabled(false);
  nicla::enableCharging(100);

  // Initialize sensors
  BHY2.begin();
  gyro.begin();
  accel.begin();
  mag.begin();
  quat.begin();
  pressure.begin();
  linAccel.begin();
  
  // Initialize BLE (Bluetooth Low Energy)
  BLE.begin();
  BLE.setLocalName("NiclaSenseME-CSV");
  BLE.setAdvertisedService(uartService);
  
  uartService.addCharacteristic(txChar);
  BLE.addService(uartService);
  
  // Set connection handlers
  BLE.setEventHandler(BLEConnected, [](BLEDevice central) {
    connected = true;
    nicla::leds.setColor(blue);
    
    // Send header
    String header = "Timestamp,Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z,Mag_X,Mag_Y,Mag_Z,Pressure,Quat_W,Quat_X,Quat_Y,Quat_Z,Lin_Acc_X,Lin_Acc_Y,Lin_Acc_Z\n";
    txChar.writeValue(header.c_str());
  });
  
  BLE.setEventHandler(BLEDisconnected, [](BLEDevice central) {
    connected = false;
    nicla::leds.setColor(green);
  });
  
  BLE.advertise();
  nicla::leds.setColor(green);
}

void loop() {
  BLEDevice central = BLE.central();
  
  if (central && central.connected()) {
    static unsigned long lastRead = 0;
    
    // Send data every 20ms (50Hz)
    if (millis() - lastRead >= 20) {
      lastRead = millis();
      
      // Update sensors
      BHY2.update();
      
      // Build CSV line
      String data = String(millis()) + "," +
                   gyro.x() + "," + gyro.y() + "," + gyro.z() + "," +
                   accel.x() + "," + accel.y() + "," + accel.z() + "," +
                   mag.x() + "," + mag.y() + "," + mag.z() + "," +
                   pressure.value() + "," +
                   quat.w() + "," + quat.x() + "," + quat.y() + "," + quat.z() + "," +
                   linAccel.x() + "," + linAccel.y() + "," + linAccel.z() + "\n";
      
      // Send over Bluetooth
      txChar.writeValue(data.c_str());
    }
  }
}