#!/usr/bin/env python3
"""
Setup für 6-Klassen Tischtennisschlag-Klassifikation
"""

import os

def setup():
    """Erstellt alle notwendigen Ordner"""
    
    # Datenordner für 6 Schlagtypen
    stroke_types = [
        'vorhand_topspin', 'vorhand_schupf', 'vorhand_block',
        'rueckhand_topspin', 'rueckhand_schupf', 'rueckhand_block'
    ]
    
    # Alle Ordner erstellen
    folders = (
        ['rohdaten/' + s for s in stroke_types] +
        ['models', 'processed_data', 'visualizations']
    )
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("Setup abgeschlossen - 6 Schlagtypen bereit")

if __name__ == "__main__":
    setup()
