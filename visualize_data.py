"""
Visualisierung von Sensordaten für Tischtennisschläge
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_single_file(filepath, save_path=None):
    """Plottet Sensordaten aus einer einzelnen CSV-Datei"""
    # Daten laden
    df = pd.read_csv(filepath)
    df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Zeit in Sekunden umrechnen
    df['time'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1000
    
    # Plot erstellen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Beschleunigung
    ax1.plot(df['time'], df['acc_x'], label='X', alpha=0.8)
    ax1.plot(df['time'], df['acc_y'], label='Y', alpha=0.8)
    ax1.plot(df['time'], df['acc_z'], label='Z', alpha=0.8)
    ax1.set_ylabel('Beschleunigung (g)')
    ax1.set_title(f'Sensordaten: {os.path.basename(filepath)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Winkelgeschwindigkeit
    ax2.plot(df['time'], df['gyro_x'], label='X', alpha=0.8)
    ax2.plot(df['time'], df['gyro_y'], label='Y', alpha=0.8)
    ax2.plot(df['time'], df['gyro_z'], label='Z', alpha=0.8)
    ax2.set_ylabel('Winkelgeschwindigkeit (°/s)')
    ax2.set_xlabel('Zeit (s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_stroke_comparison():
    """Vergleicht typische Muster verschiedener Schlagarten"""
    stroke_types = ['vorhand_topspin', 'vorhand_schupf', 'rueckhand_topspin', 'rueckhand_schupf']
    colors = ['red', 'blue', 'green', 'orange']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (stroke_type, color) in enumerate(zip(stroke_types, colors)):
        # Erste Datei des Typs laden
        files = glob.glob(f'rohdaten/{stroke_type}/*.csv')
        if files:
            df = pd.read_csv(files[0])
            df.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            
            # Magnitude berechnen
            acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            gyro_mag = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
            
            # Zeit normalisieren
            time = np.linspace(0, 1, len(df))
            
            # Plotten
            ax = axes[idx]
            ax2 = ax.twinx()
            
            line1 = ax.plot(time, acc_mag, color=color, label='Beschleunigung', linewidth=2)
            line2 = ax2.plot(time, gyro_mag, color=color, linestyle='--', 
                           label='Winkelgeschw.', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Normalisierte Zeit')
            ax.set_ylabel('Beschleunigung (g)', color=color)
            ax2.set_ylabel('Winkelgeschwindigkeit (°/s)', color=color)
            ax.set_title(stroke_type.replace('_', ' ').title())
            
            # Legende
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vergleich der Schlagarten - Typische Muster', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/stroke_comparison.png')
    plt.show()

def plot_data_distribution():
    """Zeigt die Verteilung der gesammelten Daten"""
    stroke_types = ['vorhand_topspin', 'vorhand_schupf', 'rueckhand_topspin', 'rueckhand_schupf']
    counts = []
    
    for stroke_type in stroke_types:
        files = glob.glob(f'rohdaten/{stroke_type}/*.csv')
        counts.append(len(files))
    
    # Bar Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stroke_types, counts, color=['red', 'blue', 'green', 'orange'])
    
    # Werte auf Balken anzeigen
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    plt.xlabel('Schlagart')
    plt.ylabel('Anzahl Aufnahmen')
    plt.title('Verteilung der Trainingsdaten')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/data_distribution.png')
    plt.show()

def main():
    """Hauptfunktion für Visualisierungen"""
    import sys
    
    if len(sys.argv) > 1:
        # Einzelne Datei visualisieren
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            plot_single_file(filepath)
        else:
            print(f"Datei nicht gefunden: {filepath}")
    else:
        # Alle Visualisierungen
        print("Erstelle Visualisierungen...")
        
        # Datenverteilung
        plot_data_distribution()
        
        # Schlagvergleich (wenn Daten vorhanden)
        if any(glob.glob(f'rohdaten/*/*.csv')):
            plot_stroke_comparison()
        else:
            print("Keine Daten gefunden! Bitte erst Daten aufnehmen.")

if __name__ == "__main__":
    main()
