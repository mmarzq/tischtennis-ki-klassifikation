"""
Setup-Skript für das Tischtennis-KI Projekt
Erstellt die notwendige Umgebung und überprüft die Installation
"""

import os
import sys
import subprocess

def create_virtual_env():
    """Erstellt eine virtuelle Python-Umgebung"""
    print("Erstelle virtuelle Umgebung...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    print("✓ Virtuelle Umgebung erstellt")
    
    # Aktivierungsbefehl anzeigen
    if sys.platform == "win32":
        activate = "venv\\Scripts\\activate"
    else:
        activate = "source venv/bin/activate"
    
    print(f"\nBitte aktivieren Sie die Umgebung mit:")
    print(f"  {activate}")
    print("\nDann installieren Sie die Abhängigkeiten mit:")
    print("  pip install -r requirements.txt")

def check_directories():
    """Überprüft ob alle notwendigen Ordner existieren"""
    required_dirs = [
        'rohdaten/vorhand_topspin',
        'rohdaten/vorhand_schupf',
        'rohdaten/rueckhand_topspin',
        'rohdaten/rueckhand_schupf',
        'processed_data',
        'models',
        'visualizations',
        'sensor_recording'
    ]
    
    print("\nÜberprüfe Ordnerstruktur...")
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} fehlt!")
            all_exist = False
    
    if all_exist:
        print("\n✓ Alle Ordner vorhanden!")
    else:
        print("\n✗ Einige Ordner fehlen. Bitte Projektstruktur überprüfen.")

def main():
    print("=== Tischtennis-KI Projekt Setup ===\n")
    
    # Python Version prüfen
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warnung: Python 3.8 oder höher wird empfohlen!")
    
    # Ordnerstruktur prüfen
    check_directories()
    
    # Virtuelle Umgebung anbieten
    if not os.path.exists('venv'):
        print("\nKeine virtuelle Umgebung gefunden.")
        response = input("Möchten Sie eine erstellen? (j/n): ")
        if response.lower() == 'j':
            create_virtual_env()
    else:
        print("\n✓ Virtuelle Umgebung existiert bereits")
    
    print("\n=== Nächste Schritte ===")
    print("1. Virtuelle Umgebung aktivieren")
    print("2. Abhängigkeiten installieren: pip install -r requirements.txt")
    print("3. Arduino-Sketch hochladen")
    print("4. Test ausführen: python test_installation.py")
    print("5. Mit Datenerfassung beginnen!")

if __name__ == "__main__":
    main()
