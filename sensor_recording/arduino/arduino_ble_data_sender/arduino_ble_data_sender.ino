/*
Arduino Nicla Sense ME - Optimized Bluetooth Sensor Export
High-speed data transmission for table tennis classification
Target: 50Hz data rate
*/

#include "Nicla_System.h"
#include "Arduino_BHY2.h"
#include <ArduinoBLE.h>

// Nordic UART Service
BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLECharacteristic rxChar("6E400002-B5A3-F393-E0A9-E50E24DCCA9E", BLEWrite, 20);
BLECharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E", BLENotify, 128); // BLE Paketgröße, 128 Byte für max 128 Charakter, 20 Byte wird 120 Charakter als 6 Pakete gesendet (uneffizient)

// Sensors
SensorXYZ gyroscope(SENSOR_ID_GYRO);
SensorXYZ accelerometer(SENSOR_ID_ACC);
SensorXYZ magnetometer(SENSOR_ID_MAG);
Sensor pressure(SENSOR_ID_BARO);
SensorQuaternion quaternion(SENSOR_ID_RV);
SensorXYZ linearAccel(SENSOR_ID_LACC);

// Conversion factors
const float GYRO_SCALE = 1.0 / 16.384;
const float ACCEL_SCALE = 1.0 / 4096.0;

// Status variables
bool deviceConnected = false;
unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 20; // 20ms = 50Hz


void blePeripheralConnectHandler(BLEDevice central) {
  deviceConnected = true;
  nicla::leds.setColor(blue);
  Serial.println("Connected: " + String(central.address()));
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  deviceConnected = false;
  nicla::leds.setColor(green);
  Serial.println("Disconnected");
}

void setup() {
  Serial.begin(115200);
  
  // Nicla System init
  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(red);
  
  // Battery setup
  nicla::setBatteryNTCEnabled(false);
  nicla::enableCharging(100);
  
  // BHY2 init
  BHY2.begin();
  
  // Start sensors with higher rates
  gyroscope.begin(100);     // 100 Hz
  accelerometer.begin(100); // 100 Hz
  magnetometer.begin(50);   // 50 Hz
  quaternion.begin(50);     // 50 Hz
  pressure.begin(25);       // 25 Hz
  linearAccel.begin(100);   // 100 Hz
  
  // BLE init
  if (!BLE.begin()) {
    Serial.println("BLE failed!");
    while (1) delay(100);
  }
  
  // Configure BLE for speed
  //BLE.setConnectionInterval(6, 16); // 7.5ms - 20ms (min, max in units of 1.25ms)
  BLE.setLocalName("NiclaSenseME");
  BLE.setAdvertisedService(uartService);
  
  uartService.addCharacteristic(rxChar);
  uartService.addCharacteristic(txChar);
  BLE.addService(uartService);
  
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);
  
  BLE.advertise();
  nicla::leds.setColor(green);
}

void loop() {
  BLEDevice central = BLE.central();
  
  if (central && central.connected()) {
    // Sensordaten aktualisieren
    BHY2.update();
    
    // CSV-String zusammenbauen - IDENTISCH zum USB-Code
    String csvLine = "";
    csvLine += String(millis());           // Timestamp
    csvLine += ",";
    csvLine += String(gyroscope.x());      // Gyro X
    csvLine += ",";
    csvLine += String(gyroscope.y());      // Gyro Y
    csvLine += ",";
    csvLine += String(gyroscope.z());      // Gyro Z
    csvLine += ",";
    csvLine += String(accelerometer.x());  // Acc X
    csvLine += ",";
    csvLine += String(accelerometer.y());  // Acc Y
    csvLine += ",";
    csvLine += String(accelerometer.z());  // Acc Z
    csvLine += ",";
    csvLine += String(magnetometer.x());   // Mag X
    csvLine += ",";
    csvLine += String(magnetometer.y());   // Mag Y
    csvLine += ",";
    csvLine += String(magnetometer.z());   // Mag Z
    csvLine += ",";
    csvLine += String(pressure.value());   // Pressure
    csvLine += ",";
    csvLine += String(quaternion.w());     // Quat W
    csvLine += ",";
    csvLine += String(quaternion.x());     // Quat X
    csvLine += ",";
    csvLine += String(quaternion.y());     // Quat Y
    csvLine += ",";
    csvLine += String(quaternion.z());     // Quat Z
    csvLine += ",";
    csvLine += String(linearAccel.x());    // Linear Accel X
    csvLine += ",";
    csvLine += String(linearAccel.y());    // Linear Accel Y
    csvLine += ",";
    csvLine += String(linearAccel.z());    // Linear Accel Z
    
    // Newline hinzufügen
    csvLine += "\n";
    
    // Daten über Bluetooth senden
    txChar.writeValue(csvLine.c_str());

    delay(5); 

  } else {
    // Blink when not connected
    nicla::leds.setColor(green);
    delay(1000);
    nicla::leds.setColor(off);
    delay(1000);
  }
}
