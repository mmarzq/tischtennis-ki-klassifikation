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

// Konvertierungsfaktoren basierend auf BHI260AP Datenblatt
// Seite 149: Accelerometer output = raw_value / 4096 * g
// Seite 148: Gyroscope output = raw_value / 16.384 * dps (für ±2000°/s range)
const float GYRO_SCALE = 1.0 / 16.384;     // Convert to °/s (±2000°/s range) = 2000 / 32768
const float ACCEL_SCALE = 1.0 / 4096.0;    // Convert to g (±8g range) = 8 /32768

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
  /*
  gyro.begin(100);         // 100 Hz
  accel.begin(100);        // 100 Hz
  mag.begin(50);           // 50 Hz
  quat.begin(50);          // 50 Hz
  pressure.begin(50);      // 50 Hz
  linAccel.begin(100);     // 100 Hz (wichtig für Bewegungsanalyse)
  */

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
    
    /*
    // Send header
    String header = "Timestamp,Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z,Mag_X,Mag_Y,Mag_Z,Pressure,Quat_W,Quat_X,Quat_Y,Quat_Z,Lin_Acc_X,Lin_Acc_Y,Lin_Acc_Z\n";
    txChar.writeValue(header.c_str());
    */
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
      
      // Convert sensor values to proper units
      float gyro_x_dps = gyro.x() * GYRO_SCALE;
      float gyro_y_dps = gyro.y() * GYRO_SCALE;
      float gyro_z_dps = gyro.z() * GYRO_SCALE;
      
      float accel_x_g = accel.x() * ACCEL_SCALE;
      float accel_y_g = accel.y() * ACCEL_SCALE;
      float accel_z_g = accel.z() * ACCEL_SCALE;
      
      float lin_acc_x_g = linAccel.x() * ACCEL_SCALE;
      float lin_acc_y_g = linAccel.y() * ACCEL_SCALE;
      float lin_acc_z_g = linAccel.z() * ACCEL_SCALE;
      
      // Build CSV line with converted values
      String data = String(millis()) + "," +
                   gyro_x_dps + "," + gyro_y_dps + "," + gyro_z_dps + "," +
                   accel_x_g + "," + accel_y_g + "," + accel_z_g + "," +
                   mag.x() + "," + mag.y() + "," + mag.z() + "," +
                   pressure.value() + "," +
                   quat.w() + "," + quat.x() + "," + quat.y() + "," + quat.z() + "," +
                   lin_acc_x_g + "," + lin_acc_y_g + "," + lin_acc_z_g + "\n";

      // Send over Bluetooth
      txChar.writeValue(data.c_str());
    }
  }
}