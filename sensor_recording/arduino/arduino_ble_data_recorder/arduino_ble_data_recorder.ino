/*
Arduino Nicla Sense ME - Bluetooth Sensor Data Export
Sendet identische Daten wie das USB-Skript, aber über Bluetooth

Basiert auf dem Nordic UART Service für maximale Kompatibilität
*/

#include "Nicla_System.h"
#include "Arduino_BHY2.h"
#include <ArduinoBLE.h>

// Nordic UART Service für Serial-Terminal-Kompatibilität
BLEService uartService("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
BLECharacteristic rxChar("6E400002-B5A3-F393-E0A9-E50E24DCCA9E", BLEWrite, 20);
BLECharacteristic txChar("6E400003-B5A3-F393-E0A9-E50E24DCCA9E", BLENotify, 200);

// Sensoren definieren - IDENTISCH zum USB-Code
SensorXYZ gyroscope(SENSOR_ID_GYRO);
SensorXYZ accelerometer(SENSOR_ID_ACC);
SensorXYZ magnetometer(SENSOR_ID_MAG);      // Magnetometer
Sensor pressure(SENSOR_ID_BARO);            // Barometer/Pressure
SensorQuaternion quaternion(SENSOR_ID_RV);  // Rotation Vector (Quaternion)

// Status-Variablen
bool deviceConnected = false;
unsigned long lastSensorRead = 0;
const unsigned long SENSOR_INTERVAL = 20; // 20ms = 50Hz (wie USB-Version)

// Connection Handler
void blePeripheralConnectHandler(BLEDevice central) {
  deviceConnected = true;
  nicla::leds.setColor(blue);
  Serial.println("Bluetooth-Client verbunden: " + String(central.address()));
  
  // Header senden (identisch zum USB-Code)
  String header = "Timestamp,Gyro_X,Gyro_Y,Gyro_Z,Acc_X,Acc_Y,Acc_Z,Mag_X,Mag_Y,Mag_Z,Pressure,Quat_W,Quat_X,Quat_Y,Quat_Z\n";
  txChar.writeValue(header.c_str());
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  deviceConnected = false;
  nicla::leds.setColor(red);
  Serial.println("Bluetooth-Client getrennt: " + String(central.address()));
  delay(1000);
  nicla::leds.setColor(green); // Bereit für neue Verbindung
}

void setup() {
  // Serial für Debugging
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Warte max 3 Sekunden
  
  Serial.println("=================================");
  Serial.println("Nicla Sense ME - Bluetooth Export");
  Serial.println("=================================");
  
  // Nicla System initialisieren
  nicla::begin();
  nicla::leds.begin();
  nicla::leds.setColor(red); // Start: Rot
  
  // Battery-Charging setup
  nicla::setBatteryNTCEnabled(false);
  nicla::enableCharging(100);
  
  Serial.println("Initialisiere Sensoren...");
  
  // BHY2 initialisieren
  BHY2.begin();
  
  // Sensoren starten - IDENTISCH zum USB-Code
  gyroscope.begin();
  accelerometer.begin();
  magnetometer.begin();
  quaternion.begin();
  pressure.begin();
  
  Serial.println("Sensoren erfolgreich initialisiert!");
  
  // BLE initialisieren
  Serial.println("Starte Bluetooth...");
  if (!BLE.begin()) {
    Serial.println("FEHLER: Bluetooth-Start fehlgeschlagen!");
    nicla::leds.setColor(red);
    while (1) {
      delay(100);
    }
  }
  
  // BLE-Gerätename setzen
  String deviceName = "NiclaSenseME-CSV";
  BLE.setLocalName(deviceName.c_str());
  BLE.setAdvertisedService(uartService);
  
  // UART-Service konfigurieren
  uartService.addCharacteristic(rxChar);
  uartService.addCharacteristic(txChar);
  BLE.addService(uartService);
  
  // Event Handler setzen
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);
  
  // Advertising starten
  BLE.advertise();
  
  nicla::leds.setColor(green); // Bereit: Grün
  
  Serial.println("Bluetooth bereit!");
  Serial.println("Gerätename: " + deviceName);
  Serial.println("Warte auf Verbindung...");
  Serial.println("=================================");
}

void loop() {
  // Warte auf BLE-Verbindung und sende Daten
  BLEDevice central = BLE.central();
  
  if (central && central.connected()) {
    // Sensordaten alle SENSOR_INTERVAL Millisekunden senden
    if (millis() - lastSensorRead >= SENSOR_INTERVAL) {
      lastSensorRead = millis();
      
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
      
      // Newline hinzufügen
      csvLine += "\n";
      
      // Daten über Bluetooth senden
      txChar.writeValue(csvLine.c_str());
      
      // Debug: Erste 5 Zeilen auch an Serial
      static int debugCount = 0;
      if (debugCount < 5) {
        Serial.print("BT> ");
        Serial.print(csvLine);
        debugCount++;
      }
    }
  } else {
    // Nicht verbunden - LED blinken lassen
    static unsigned long lastBlink = 0;
    static bool ledState = false;
    
    if (millis() - lastBlink > 1000) {
      lastBlink = millis();
      if (ledState) {
        nicla::leds.setColor(green);
      } else {
        nicla::leds.setColor(off);
      }
      ledState = !ledState;
    }
  }
}

// bluetoothPrint Funktion wird nicht mehr benötigt, da wir direkt writeValue verwenden
