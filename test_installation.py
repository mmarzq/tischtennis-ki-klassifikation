"""
Test-Skript zur Überprüfung der Installation
"""

import sys
import importlib.util

def check_package(package_name):
    """Überprüft ob ein Package installiert ist"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False
    return True

def main():
    print("=== Tischtennis-KI Installation Check ===\n")
    
    # Wichtige Packages prüfen
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow',
        'serial': 'PySerial',
        'scipy': 'SciPy'
    }
    
    all_installed = True
    
    for package, name in packages.items():
        if check_package(package):
            print(f"✓ {name} ist installiert")
        else:
            print(f"✗ {name} fehlt!")
            all_installed = False
    
    print("\n" + "="*40)
    
    if all_installed:
        print("✓ Alle Packages sind installiert!")
        print("\nSie können jetzt mit der Datenerfassung beginnen:")
        print("  cd sensor_recording")
        print("  python arduino_recorder.py")
    else:
        print("✗ Einige Packages fehlen!")
        print("\nBitte installieren Sie alle Abhängigkeiten:")
        print("  pip install -r requirements.txt")
    
    # Python Version prüfen
    print(f"\nPython Version: {sys.version}")
    if sys.version_info >= (3, 8):
        print("✓ Python Version ist kompatibel")
    else:
        print("✗ Python 3.8 oder höher wird empfohlen")

if __name__ == "__main__":
    main()
