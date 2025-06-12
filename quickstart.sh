#!/bin/bash

echo "==================================="
echo "Tischtennis-KI Projekt Quickstart"
echo "==================================="
echo

# Prüfe ob Python installiert ist
if ! command -v python3 &> /dev/null; then
    echo "FEHLER: Python3 ist nicht installiert!"
    echo "Bitte installieren Sie Python 3.8 oder höher."
    exit 1
fi

# Prüfe ob venv existiert
if [ ! -d "venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

# Aktiviere venv
echo "Aktiviere virtuelle Umgebung..."
source venv/bin/activate

# Installiere Abhängigkeiten wenn nötig
if ! pip show numpy &> /dev/null; then
    echo "Installiere Abhängigkeiten..."
    pip install -r requirements.txt
fi

# Menü-Funktion
show_menu() {
    clear
    echo
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║     Tischtennisschlag-Klassifikation mit KI          ║"
    echo "║                  QUICKSTART                           ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo
    echo "Was möchten Sie tun?"
    echo
    echo "1) Sensor-Datenaufnahme starten"
    echo "2) Installation testen"
    echo "3) Daten visualisieren"
    echo "4) Modell trainieren"
    echo "5) Echtzeit-Vorhersage"
    echo "6) Beenden"
    echo
    read -p "Ihre Wahl (1-6): " choice
}

# Hauptschleife
while true; do
    show_menu
    
    case $choice in
        1)
            cd sensor_recording
            python master_recorder.py
            cd ..
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
        2)
            python test_installation.py
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
        3)
            python visualize_data.py
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
        4)
            echo
            echo "Starte Training..."
            echo "Dies kann einige Minuten dauern."
            echo
            python data_preprocessing.py
            if [ $? -eq 0 ]; then
                python train_1d_cnn.py
            fi
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
        5)
            python realtime_prediction.py
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
        6)
            deactivate
            echo
            echo "Auf Wiedersehen!"
            exit 0
            ;;
        *)
            echo "Ungültige Eingabe!"
            read -p "Drücken Sie Enter zum Fortfahren..."
            ;;
    esac
done
