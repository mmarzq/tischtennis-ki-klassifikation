# Tischtennisschlag-Klassifikation mit 1D CNN

Dieses Projekt verwendet IMU-Sensoren (Arduino Nicla Sense ME / X-IO NGIMU) und ein 1D Convolutional Neural Network zur automatischen Klassifikation von Tischtennisschlägen.

## 📋 Übersicht

Das System kann folgende Schlagarten unterscheiden:
- Vorhand Topspin
- Vorhand Schupf
- Rückhand Topspin
- Rückhand Schupf

## 🎯 Unterstützte Sensoren

### Arduino Nicla Sense ME
- **Verbindung**: USB oder Bluetooth
- **Sensoren**: BMI270 (Acc/Gyro) + BMM150 (Mag)
- **Vorteile**: Einfache Einrichtung, günstig, kompakt
- **Ideal für**: Erste Tests, Prototyping

### X-IO NGIMU
- **Verbindung**: WiFi oder USB
- **Sensoren**: InvenSense MPU-9250
- **Vorteile**: Hohe Präzision, viele Features, SD-Karte
- **Ideal für**: Professionelle Datenerfassung

## 🚀 Installation

### 1. Repository verwenden
```bash
cd C:\Users\labib\Workspace\tischtennis-ki-klassifikation
```

### 2. Python-Umgebung einrichten
```bash
# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 3. Sensor vorbereiten

#### Für Arduino Nicla Sense ME:
1. Arduino IDE installieren
2. Board Package "Arduino Mbed OS Nicla Boards" installieren
3. Bibliothek "Arduino_BHY2" installieren
4. Sketch `sensor_recording/nicla_sense_me_recorder.ino` auf Arduino laden

#### Für X-IO NGIMU:
1. NGIMU GUI von [x-io.co.uk](https://x-io.co.uk/ngimu/) herunterladen
2. NGIMU einschalten und mit Netzwerk verbinden
3. In GUI: Send Rates konfigurieren (Sensors: 100Hz)
4. OSC-Ausgabe aktivieren

## 📊 Workflow

### 1. Datenerfassung

#### Schnellstart (Master Recorder):
```bash
# Windows
quickstart.bat

# Linux/Mac
chmod +x quickstart.sh
./quickstart.sh

# Oder direkt:
cd sensor_recording
python master_recorder.py
```

Der **Master Recorder** bietet ein interaktives Menü mit folgenden Optionen:
- **Arduino-Datenaufnahme**: Direkte Aufnahme von Sensordaten
- **NGIMU-Datenaufnahme**: OSC-basierte Aufnahme mit verschiedenen Modi
- **Sensor-Vergleich**: Vergleich beider Sensoren in Echtzeit
- **Datenvorverarbeitung**: Direkt aus dem Menü starten
- **Visualisierung**: Daten direkt visualisieren

#### Arduino-spezifische Aufnahme:
```bash
cd sensor_recording
python arduino_recorder.py
```

#### NGIMU-spezifische Tools:
```bash
cd sensor_recording/xio_ngimu

# Sensor konfigurieren
python ngimu_configurator.py

# OSC-Datenaufnahme
python ngimu_osc_recorder.py

# Echtzeit-Visualisierung
python ngimu_realtime_visualizer.py

# CSV-Konvertierung
python ngimu_csv_converter.py
```

#### Sensor-Vergleich:
```bash
cd sensor_recording
python sensor_comparison.py
```

Dieses Tool ermöglicht:
- Gleichzeitige Aufnahme beider Sensoren
- Synchronisierte Visualisierung
- Performance-Vergleich in Echtzeit

### 2. Datenvorverarbeitung

```bash
python data_preprocessing.py
```

Dies:
- Lädt alle Rohdaten
- Erkennt einzelne Schläge automatisch
- Segmentiert die Daten in Fenster
- Speichert vorverarbeitete Daten in `processed_data/`

### 3. Modell trainieren

```bash
python train_1d_cnn.py
```

Das Training:
- Lädt die vorverarbeiteten Daten
- Trainiert ein 1D CNN Modell
- Zeigt Trainingsfortschritt und Metriken
- Speichert das beste Modell in `models/`

### 4. Echtzeit-Vorhersage

```bash
# Mit echtem Sensor
python realtime_prediction.py

# Demo-Modus (ohne Sensor)
python realtime_prediction.py --demo
```

## 📁 Projektstruktur

```
tischtennis-ki-klassifikation/
├── rohdaten/                    # Rohe Sensordaten
│   ├── vorhand_topspin/
│   ├── vorhand_schupf/
│   ├── rueckhand_topspin/
│   └── rueckhand_schupf/
├── processed_data/              # Vorverarbeitete Daten
├── models/                      # Trainierte Modelle
├── visualizations/              # Grafiken und Plots
├── sensor_recording/            # Datenerfassungs-Skripte
│   ├── master_recorder.py       # Hauptmenü für alle Aufnahmen
│   ├── arduino_recorder.py      # Arduino-spezifische Aufnahme
│   ├── sensor_comparison.py     # Sensor-Vergleichstool
│   ├── nicla_sense_me_recorder.ino  # Arduino-Sketch
│   └── xio_ngimu/              # NGIMU-spezifische Tools
│       ├── ngimu_configurator.py    # Sensor-Konfiguration
│       ├── ngimu_osc_recorder.py    # OSC-Datenaufnahme
│       ├── ngimu_realtime_visualizer.py  # Live-Visualisierung
│       └── ngimu_csv_converter.py   # CSV-Konvertierung
├── data_preprocessing.py        # Datenvorverarbeitung
├── train_1d_cnn.py             # Modelltraining
├── realtime_prediction.py       # Echtzeit-Vorhersage
├── visualize_data.py           # Datenvisualisierung
├── quickstart.bat/sh           # Schnellstart-Skripte
└── requirements.txt            # Python-Abhängigkeiten
```

## 🔧 Sensor-Konfiguration

### Arduino Nicla Sense ME
- **Abtastrate**: 100 Hz
- **Sensoren**: Accelerometer (±4g), Gyroskop (±2000°/s)
- **Verbindung**: USB (COM4) oder Bluetooth
- **Baudrate**: 115200

### X-IO NGIMU
- **Abtastrate**: 100-400 Hz
- **Sensoren**: Accelerometer (±16g), Gyroskop (±2000°/s), Magnetometer
- **Verbindung**: WiFi (OSC Port 9000) oder USB
- **Extras**: Quaternion, Lineare Beschleunigung

### Sensor-Befestigung
- Am Schlägergriff befestigen
- Konsistente Ausrichtung wichtig
- Sichere Befestigung (3D-Druck Halterung empfohlen)

## 🛠️ Erweiterte Features

### Master Recorder
Der `master_recorder.py` ist das zentrale Tool für die Datenerfassung:
- Automatische Sensor-Erkennung
- Interaktives Menü für alle Funktionen
- Direkte Integration aller Tools
- Workflow-Unterstützung

### NGIMU-Tools
Die spezialisierten NGIMU-Tools bieten:
- **Configurator**: GUI-basierte Sensor-Konfiguration
- **OSC Recorder**: Hochperformante Datenaufnahme über Netzwerk
- **Realtime Visualizer**: Live-Ansicht aller Sensordaten
- **CSV Converter**: Konvertierung zwischen verschiedenen Datenformaten

### Sensor-Vergleich
Das Vergleichstool ermöglicht:
- Synchronisierte Aufnahme beider Sensoren
- Echtzeit-Performance-Metriken
- Qualitätsvergleich der Sensordaten
- Export von Vergleichsberichten

## 📈 Modell-Details

### 1D CNN Architektur
- 3 Convolutional Blocks mit BatchNorm und Dropout
- Global Max Pooling
- 2 Dense Layers
- Softmax Output für 4 Klassen

### Performance
- Erwartete Genauigkeit: 85-95%
- Echtzeitfähig (<100ms Latenz)

## 🛠️ Troubleshooting

### "Keine Daten empfangen"
- COM-Port überprüfen (Geräte-Manager)
- Arduino-Sketch läuft?
- Baudrate stimmt überein?
- Bei NGIMU: OSC-Ausgabe aktiviert?

### "Modell nicht gefunden"
- Erst Daten sammeln
- Dann vorverarbeiten
- Dann trainieren

### Schlechte Vorhersagegenauigkeit
- Mehr Trainingsdaten sammeln (min. 50 Schläge pro Typ)
- Sensor-Befestigung überprüfen
- Verschiedene Spieler einbeziehen

### NGIMU-Verbindungsprobleme
- Firewall-Einstellungen prüfen (Port 9000)
- IP-Adresse korrekt?
- NGIMU im gleichen Netzwerk?

## 📝 Tipps für gute Ergebnisse

1. **Datenqualität**:
   - Klare, deutliche Schläge ausführen
   - Pausen zwischen Schlägen lassen
   - Verschiedene Geschwindigkeiten/Intensitäten

2. **Konsistenz**:
   - Immer gleiche Sensor-Position
   - Gleiche Ausrichtung
   - Kalibrierung vor jeder Session

3. **Vielfalt**:
   - Verschiedene Spieler
   - Verschiedene Schläger
   - Realistische Spielsituationen

## 🎯 Nächste Schritte

- [ ] Mehr Schlagarten hinzufügen (Aufschlag, Block, etc.)
- [ ] Sensor-Fusion mit Videoanalyse
- [ ] Mobile App entwickeln
- [ ] Cloud-basiertes Training
- [ ] Feedback-System für Spieler

