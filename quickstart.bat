@echo off
echo ===================================
echo Tischtennis-KI Projekt Quickstart
echo ===================================
echo.

REM Prüfe ob Python installiert ist
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python ist nicht installiert oder nicht im PATH!
    echo Bitte installieren Sie Python 3.8 oder höher.
    pause
    exit /b 1
)

REM Prüfe ob venv existiert
if not exist "venv" (
    echo Erstelle virtuelle Umgebung...
    python -m venv venv
)

REM Aktiviere venv
echo Aktiviere virtuelle Umgebung...
call venv\Scripts\activate.bat

REM Installiere Abhängigkeiten wenn nötig
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installiere Abhängigkeiten...
    pip install -r requirements.txt
)

REM Zeige Menü
:menu
cls
echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║     Tischtennisschlag-Klassifikation mit KI          ║
echo ║                  QUICKSTART                           ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
echo Was möchten Sie tun?
echo.
echo 1) Sensor-Datenaufnahme starten
echo 2) Installation testen
echo 3) Daten visualisieren
echo 4) Modell trainieren
echo 5) Echtzeit-Vorhersage
echo 6) Beenden
echo.
set /p choice="Ihre Wahl (1-6): "

if "%choice%"=="1" goto recording
if "%choice%"=="2" goto test
if "%choice%"=="3" goto visualize
if "%choice%"=="4" goto train
if "%choice%"=="5" goto predict
if "%choice%"=="6" goto end

echo Ungültige Eingabe!
pause
goto menu

:recording
cd sensor_recording
python master_recorder.py
cd ..
pause
goto menu

:test
python test_installation.py
pause
goto menu

:visualize
python visualize_data.py
pause
goto menu

:train
echo.
echo Starte Training...
echo Dies kann einige Minuten dauern.
echo.
python data_preprocessing.py
if errorlevel 0 (
    python train_1d_cnn.py
)
pause
goto menu

:predict
python realtime_prediction.py
pause
goto menu

:end
deactivate
echo.
echo Auf Wiedersehen!
pause
