# X-IO NGIMU Integration

Dieses Verzeichnis enthält alle Skripte zur Datenerfassung mit dem X-IO NGIMU Sensor.

## 📋 Übersicht der Skripte

### 1. `ngimu_osc_recorder.py`
**Echtzeit-Datenerfassung über WiFi/Ethernet**
- Empfängt Sensordaten über OSC-Protokoll
- Interaktive Aufnahmesession für verschiedene Schlagtypen
- Automatische Speicherung im einheitlichen CSV-Format

### 2. `ngimu_configurator.py`
**Sensor-Konfiguration über OSC**
- Optimierte Einstellungen für Tischtennis
- Kalibrierung und Reset-Funktionen
- Anpassung der Datenraten

### 3. `ngimu_csv_converter.py`
**Konvertierung von NGIMU GUI Exporten**
- Wandelt NGIMU CSV-Format in einheitliches Format um
- Batch-Konvertierung ganzer Ordner
- Analyse-Funktionen für CSV-Dateien

### 4. `ngimu_realtime_visualizer.py`
**Live-Visualisierung der Sensordaten**
- Echtzeit-Plots für alle Sensoren
- Bewegungsintensität und Schlagerkennung
- Visuelles Feedback während der Aufnahme

## 🚀 Schnellstart

### Voraussetzungen

1. **NGIMU einschalten** und mit dem Netzwerk verbinden
2. **Python-Abhängigkeiten** installieren:
   ```bash
   pip install python-osc
   ```

### NGIMU Konfiguration

1. **NGIMU GUI** öffnen
2. **WiFi-Einstellungen**:
   - Mode: Client oder Access Point
   - IP-Adresse notieren (z.B. 192.168.1.100)
3. **Send Rates** einstellen:
   - /rate/sensors: 100
   - /rate/quaternion: 100
   - /rate/linear: 100
4. **OSC aktivieren**:
   - Destination IP: Ihr Computer
   - Port: 9000

### Datenerfassung starten

```bash
cd sensor_recording/xio_ngimu
python ngimu_osc_recorder.py
```

## 📊 Workflow-Beispiel

### 1. Sensor konfigurieren
```bash
python ngimu_configurator.py
# IP eingeben, dann Option 1 für Tischtennis-Optimierung
```

### 2. Verbindung testen
```bash
python ngimu_realtime_visualizer.py
# Sensor bewegen und Live-Daten prüfen
```

### 3. Daten aufnehmen
```bash
python ngimu_osc_recorder.py
# Schlagtyp wählen und Aufnahme starten
```

### 4. GUI-Daten konvertieren (optional)
```bash
python ngimu_csv_converter.py
# Falls Sie Daten über NGIMU GUI aufgenommen haben
```

## 🔧 Tipps & Tricks

### Optimale NGIMU-Einstellungen

**Für beste Ergebnisse:**
- Sensors Rate: 100 Hz
- Quaternion Rate: 100 Hz  
- Linear Rate: 100 Hz (optional)
- Euler Rate: 0 Hz (nicht benötigt)
- Andere Sensoren: 0 Hz (deaktiviert)

### Netzwerk-Troubleshooting

**Wenn keine Daten ankommen:**
1. Firewall-Einstellungen prüfen (Port 9000)
2. Ping zum NGIMU: `ping 192.168.1.100`
3. In NGIMU GUI: Tools → Network Announce
4. Richtiges Subnetz? (Computer und NGIMU im gleichen Netzwerk)

### Batterie-Management

- Vollständig laden vor längeren Sessions
- Batterie-Status über GUI oder `/battery` OSC-Befehl
- Bei niedrigem Stand: Datenqualität kann leiden

## 📈 Datenformat

### OSC-Nachrichten vom NGIMU

| Adresse | Daten | Einheit | Rate |
|---------|-------|---------|------|
| /sensors | timestamp, gyroX/Y/Z, accX/Y/Z, magX/Y/Z | °/s, g, µT | 100 Hz |
| /quaternion | w, x, y, z | - | 100 Hz |
| /linear | x, y, z | g | 100 Hz |
| /euler | roll, pitch, yaw | ° | Optional |

### Konvertiertes CSV-Format

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
1234,0.12,-0.03,0.98,1.2,-0.5,0.1
```

## 🆚 Vergleich Arduino vs NGIMU

| Feature | Arduino Nicla | X-IO NGIMU |
|---------|---------------|------------|
| Verbindung | USB/Bluetooth | WiFi/USB |
| Max. Rate | ~200 Hz | 400 Hz |
| Sensoren | BMI270 + BMM150 | Invensense MPU9250 |
| Echtzeit | Mittel | Sehr gut |
| Preis | ~80€ | ~380£ |
| Setup | Einfach | Komplex |

## 📞 Support

Bei Problemen:
1. NGIMU User Manual konsultieren
2. [X-IO Forum](https://x-io.co.uk/community/)
3. Logs in NGIMU GUI prüfen
