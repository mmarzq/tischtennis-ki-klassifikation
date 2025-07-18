/*
 * Arduino Nicla Sense ME - Tischtennisschlag Datenerfassung
 * Sendet IMU-Daten über serielle Schnittstelle für KI-Training
 */

#include "Arduino_BHY2.h"

// Sensoren definieren
SensorXYZ accelerometer(SENSOR_ID_ACC);
SensorXYZ gyroscope(SENSOR_ID_GYRO);

// Timing
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL = 10; // 10ms = 100Hz

void setup() {
  // Serielle Verbindung
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  // BHY2 initialisieren
  BHY2.begin();
  
  // Sensoren starten
  accelerometer.begin();
  gyroscope.begin();
  
  // LED zur Statusanzeige
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH); // LED an = bereit
  
  delay(1000); // Stabilisierung
}

void loop() {
  // Sensordaten aktualisieren
  BHY2.update();
  
  // Sample-Rate kontrollieren
  unsigned long currentTime = millis();
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    // Daten im CSV-Format senden
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(accelerometer.x(), 3);
    Serial.print(",");
    Serial.print(accelerometer.y(), 3);
    Serial.print(",");
    Serial.print(accelerometer.z(), 3);
    Serial.print(",");
    Serial.print(gyroscope.x(), 3);
    Serial.print(",");
    Serial.print(gyroscope.y(), 3);
    Serial.print(",");
    Serial.println(gyroscope.z(), 3);
  }
}
