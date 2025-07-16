# XIO NGIMU Sensor Recording

Dieses Verzeichnis enthält Skripte zur Aufnahme und Visualisierung von Sensordaten des x-io NGIMU (Next Generation Inertial Measurement Unit) für die Tischtennisschlag-Klassifikation.

## Überblick

Das x-io NGIMU ist ein drahtloses 9-DOF IMU-System, das über WiFi kommuniziert und OSC (Open Sound Control) Nachrichten sendet. Die Skripte in diesem Verzeichnis ermöglichen:

- Aufnahme von Sensordaten (Gyroskop, Beschleunigung, Magnetometer, Quaternion, Lineare Beschleunigung)
- Echtzeit-Visualisierung der Sensordaten
- Speicherung der Daten im CSV-Format für weitere Analysen

## Dateien

### 1. `xio_ngimu_recorder.py`
Hauptskript zur Aufnahme von Sensordaten.

**Funktionen:**
- Verbindet sich über UDP mit dem NGIMU (Standard IP: 192.168.1.1)
- Empfängt Daten auf mehreren Ports (8001, 8010, 8011, 8012)
- Speichert folgende Sensordaten:
  - Gyroskop (X, Y, Z) in °/s
  - Beschleunigung (X, Y, Z) in g
  - Magnetometer (X, Y, Z)
  - Barometer in hPa
  - Quaternion (W, X, Y, Z)
  - Lineare Beschleunigung (X, Y, Z) in g (ohne Gravitation)
- Interaktives Menü für verschiedene Schlagtypen
- Automatische Dateinamensvergabe mit Zeitstempel

**Verwendung:**
```bash
python xio_ngimu_recorder.py
```

### 2. `ngimu_realtime_visualizer.py`
Echtzeit-Visualisierung der Sensordaten mit matplotlib.

**Funktionen:**
- Zeigt 4 Diagramme in Echtzeit:
  1. Rohe Beschleunigung (3 Achsen)
  2. Gyroskop (3 Achsen)
  3. Lineare Beschleunigung (3 Achsen, ohne Gravitation)
  4. Bewegungsintensität
- Visuelle Schlag-Erkennung (rötlicher Hintergrund bei hoher Intensität)
- Automatische Skalierung der Zeitachse

**Verwendung:**
```bash
python ngimu_realtime_visualizer.py
```

### 3. `osc_decoder.py`
OSC-Decoder vom Hersteller x-io Technologies.

**Funktionen:**
- Dekodiert OSC-Nachrichten vom NGIMU
- Unterstützt verschiedene Datentypen (float, int, string, bool)
- Verarbeitet OSC-Bundles und einzelne Nachrichten

## Installation

### Voraussetzungen
- Python 3.7+
- Erforderliche Pakete:
  ```bash
  pip install matplotlib numpy
  ```

### NGIMU Konfiguration
1. NGIMU über WiFi verbinden (Standard SSID: NGIMU)
2. IP-Adresse: 192.168.1.1
3. Send Port: 9000
4. Receive Ports: 8001, 8010, 8011, 8012

## Datenformat

Die CSV-Dateien enthalten folgende Spalten:
- `Timestamp`: Zeit seit Aufnahmestart in Sekunden
- `Gyro_X`, `Gyro_Y`, `Gyro_Z`: Winkelgeschwindigkeit in °/s
- `Acc_X`, `Acc_Y`, `Acc_Z`: Beschleunigung in g
- `Mag_X`, `Mag_Y`, `Mag_Z`: Magnetfeld
- `Bar`: Luftdruck in hPa
- `Quat_W`, `Quat_X`, `Quat_Y`, `Quat_Z`: Orientierung als Quaternion
- `Lin_Acc_X`, `Lin_Acc_Y`, `Lin_Acc_Z`: Lineare Beschleunigung in g (ohne Gravitation)

## Schlagtypen

Die Aufnahme unterstützt 4 Tischtennisschlag-Typen:
1. Vorhand Topspin
2. Vorhand Schupf
3. Rückhand Topspin
4. Rückhand Schupf

## Tipps zur Verwendung

1. **Stabile Verbindung**: Stellen Sie sicher, dass die WiFi-Verbindung zum NGIMU stabil ist
2. **Aufnahmequalität**: Das System erreicht typischerweise 100-400 Hz Abtastrate
3. **Synchronisation**: Die Skripte warten auf vollständige Datensätze (Sensoren + Quaternion + Lineare Beschleunigung) bevor sie speichern
4. **Visualisierung**: Nutzen Sie den Visualizer zur Überprüfung der Datenqualität in Echtzeit

## Troubleshooting

- **Keine Daten empfangen**: Überprüfen Sie die WiFi-Verbindung und IP-Adresse
- **Niedrige Datenrate**: Stellen Sie sicher, dass alle Ports (8001-8012) nicht blockiert sind
- **Fehlende Daten**: Der NGIMU muss konfiguriert sein, um alle Datentypen zu senden

## Lizenz

Der `osc_decoder.py` ist Copyright © x-io Technologies Limited.
Die anderen Skripte sind Teil des Tischtennisschlag-Klassifikationsprojekts.
