import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns

def plot_all_csv_files():
    """Plot all CSV files in all stroke type folders automatically"""
    
    # Set up folder paths
    base_folder = "../../rohdaten"
    stroke_types = {
        'vorhand_topspin': 'Vorhand Topspin',
        'vorhand_schupf': 'Vorhand Schupf',
        'rueckhand_topspin': 'Rückhand Topspin',
        'rueckhand_schupf': 'Rückhand Schupf'
    }
    
    print("\n=== XIO NGIMU Tischtennisschlag Visualisierung ===")
    print("Verarbeite alle CSV-Dateien in allen Unterordnern...\n")
    
    total_files_processed = 0
    
    # Process each stroke type folder
    for folder_name, stroke_name in stroke_types.items():
        input_folder = os.path.join(base_folder, folder_name)
        output_folder = os.path.join(base_folder, f"{folder_name}_plots")
        
        # Check if input folder exists
        if not os.path.exists(input_folder):
            print(f"⚠️  Ordner nicht gefunden: {input_folder}")
            continue
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all CSV files in the input folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"⚠️  Keine CSV-Dateien in {input_folder} gefunden!")
            continue
        
        print(f"\n📁 {stroke_name}:")
        print(f"   Gefundene CSV-Dateien: {len(csv_files)}")
        
        # Process each CSV file
        for i, csv_file in enumerate(csv_files, 1):
            file_path = os.path.join(input_folder, csv_file)
            output_path = os.path.join(output_folder, csv_file.replace('.csv', '_plot.png'))
            
            print(f"   [{i}/{len(csv_files)}] Verarbeite: {csv_file}", end="")
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Create comprehensive visualization
                create_imu_visualization(df, output_path, csv_file, stroke_name)
                
                print(" ✓")
                total_files_processed += 1
                
            except Exception as e:
                print(f" ✗ Fehler: {e}")
        
        print(f"   Plots gespeichert in: {output_folder}")
    
    print(f"\n✅ Fertig! Insgesamt {total_files_processed} Dateien verarbeitet.")
    
    # Create comparison plots
    print("\n📊 Erstelle Vergleichsplots...")
    create_comparison_plots(base_folder)

def create_imu_visualization(df, output_path, filename, stroke_type):
    """Create comprehensive IMU data visualization"""
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Title
    fig.suptitle(f'IMU Datenanalyse - {stroke_type}\n{filename}', 
                 fontsize=16, fontweight='bold')
    
    # Color schemes
    gyro_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    acc_colors = ['#FFA07A', '#98D8C8', '#6C5CE7']
    mag_colors = ['#FFEAA7', '#81C784', '#9575CD']
    quat_colors = ['#FF7675', '#74B9FF', '#A29BFE', '#FD79A8']
    
    # 1. Gyroscope data (3 axes)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Timestamp'], df['Gyro_X'], color=gyro_colors[0], label='Gyro X', linewidth=1.5)
    ax1.plot(df['Timestamp'], df['Gyro_Y'], color=gyro_colors[1], label='Gyro Y', linewidth=1.5)
    ax1.plot(df['Timestamp'], df['Gyro_Z'], color=gyro_colors[2], label='Gyro Z', linewidth=1.5)
    ax1.set_title('Gyroskop Daten (rad/s)', fontweight='bold')
    ax1.set_xlabel('Zeit (s)')
    ax1.set_ylabel('Winkelgeschwindigkeit (rad/s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Accelerometer data (3 axes)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df['Timestamp'], df['Acc_X'], color=acc_colors[0], label='Acc X', linewidth=1.5)
    ax2.plot(df['Timestamp'], df['Acc_Y'], color=acc_colors[1], label='Acc Y', linewidth=1.5)
    ax2.plot(df['Timestamp'], df['Acc_Z'], color=acc_colors[2], label='Acc Z', linewidth=1.5)
    ax2.set_title('Beschleunigungssensor Daten (m/s²)', fontweight='bold')
    ax2.set_xlabel('Zeit (s)')
    ax2.set_ylabel('Beschleunigung (m/s²)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Magnetometer data (3 axes)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(df['Timestamp'], df['Mag_X'], color=mag_colors[0], label='Mag X', linewidth=1.5)
    ax3.plot(df['Timestamp'], df['Mag_Y'], color=mag_colors[1], label='Mag Y', linewidth=1.5)
    ax3.plot(df['Timestamp'], df['Mag_Z'], color=mag_colors[2], label='Mag Z', linewidth=1.5)
    ax3.set_title('Magnetometer Daten (μT)', fontweight='bold')
    ax3.set_xlabel('Zeit (s)')
    ax3.set_ylabel('Magnetfeld (μT)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Quaternion data
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(df['Timestamp'], df['Quat_W'], color=quat_colors[0], label='Quat W', linewidth=1.5)
    ax4.plot(df['Timestamp'], df['Quat_X'], color=quat_colors[1], label='Quat X', linewidth=1.5)
    ax4.plot(df['Timestamp'], df['Quat_Y'], color=quat_colors[2], label='Quat Y', linewidth=1.5)
    ax4.plot(df['Timestamp'], df['Quat_Z'], color=quat_colors[3], label='Quat Z', linewidth=1.5)
    ax4.set_title('Quaternion (Orientierung)', fontweight='bold')
    ax4.set_xlabel('Zeit (s)')
    ax4.set_ylabel('Quaternion Werte')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Linear Acceleration
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.plot(df['Timestamp'], df['Lin_Acc_X'], color='#E74C3C', label='Lin Acc X', linewidth=1.5)
    ax5.plot(df['Timestamp'], df['Lin_Acc_Y'], color='#3498DB', label='Lin Acc Y', linewidth=1.5)
    ax5.plot(df['Timestamp'], df['Lin_Acc_Z'], color='#2ECC71', label='Lin Acc Z', linewidth=1.5)
    ax5.set_title('Lineare Beschleunigung (ohne Gravitation)', fontweight='bold')
    ax5.set_xlabel('Zeit (s)')
    ax5.set_ylabel('Beschleunigung (m/s²)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Barometer
    ax6 = fig.add_subplot(gs[3, 2])
    ax6.plot(df['Timestamp'], df['Bar'], color='#9B59B6', linewidth=2)
    ax6.set_title('Luftdruck', fontweight='bold')
    ax6.set_xlabel('Zeit (s)')
    ax6.set_ylabel('Druck (hPa)')
    ax6.grid(True, alpha=0.3)
    
    # 7. Total acceleration magnitude
    ax7 = fig.add_subplot(gs[4, 0])
    acc_magnitude = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)
    ax7.plot(df['Timestamp'], acc_magnitude, color='#E67E22', linewidth=2)
    ax7.fill_between(df['Timestamp'], 0, acc_magnitude, alpha=0.3, color='#E67E22')
    ax7.set_title('Gesamtbeschleunigung (Magnitude)', fontweight='bold')
    ax7.set_xlabel('Zeit (s)')
    ax7.set_ylabel('|Acc| (m/s²)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Gyroscope magnitude
    ax8 = fig.add_subplot(gs[4, 1])
    gyro_magnitude = np.sqrt(df['Gyro_X']**2 + df['Gyro_Y']**2 + df['Gyro_Z']**2)
    ax8.plot(df['Timestamp'], gyro_magnitude, color='#16A085', linewidth=2)
    ax8.fill_between(df['Timestamp'], 0, gyro_magnitude, alpha=0.3, color='#16A085')
    ax8.set_title('Gesamt-Winkelgeschwindigkeit (Magnitude)', fontweight='bold')
    ax8.set_xlabel('Zeit (s)')
    ax8.set_ylabel('|Gyro| (rad/s)')
    ax8.grid(True, alpha=0.3)
    
    # 9. Statistics box
    ax9 = fig.add_subplot(gs[4, 2])
    ax9.axis('off')
    
    # Calculate statistics
    stats_text = f"""Statistiken:
    
Dauer: {df['Timestamp'].max():.3f} s
Samples: {len(df)}
Sample Rate: {len(df)/df['Timestamp'].max():.1f} Hz

Max Beschleunigung: {acc_magnitude.max():.2f} m/s²
Max Winkelgeschw.: {gyro_magnitude.max():.2f} rad/s

Durchschn. Luftdruck: {df['Bar'].mean():.1f} hPa
Luftdruck Bereich: {df['Bar'].min():.1f} - {df['Bar'].max():.1f} hPa
"""
    
    ax9.text(0.1, 0.5, stats_text, transform=ax9.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_plots(base_folder):
    """Create comparison plots for different stroke types"""
    stroke_types = {
        'vorhand_topspin': 'Vorhand Topspin',
        'vorhand_schupf': 'Vorhand Schupf',
        'rueckhand_topspin': 'Rückhand Topspin',
        'rueckhand_schupf': 'Rückhand Schupf'
    }
    
    # Create two comparison plots
    # 1. Acceleration magnitude comparison
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Vergleich der Schlagarten - Gesamtbeschleunigung', fontsize=16, fontweight='bold')
    
    # 2. Gyroscope magnitude comparison
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Vergleich der Schlagarten - Winkelgeschwindigkeit', fontsize=16, fontweight='bold')
    
    for idx, (folder_name, title) in enumerate(stroke_types.items()):
        ax1 = axes1[idx // 2, idx % 2]
        ax2 = axes2[idx // 2, idx % 2]
        input_folder = os.path.join(base_folder, folder_name)
        
        if os.path.exists(input_folder):
            csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
            
            # Use colormap for different files
            colors = plt.cm.viridis(np.linspace(0, 1, min(len(csv_files), 10))) #min(len(csv_files), 10)
            
            for i, csv_file in enumerate(csv_files[20:30]): #[:10] # Plot first 10 files
                file_path = os.path.join(input_folder, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Acceleration magnitude
                    acc_magnitude = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)
                    ax1.plot(df['Timestamp'], acc_magnitude, alpha=0.6, linewidth=1, 
                            color=colors[i]) #, label=f'Aufnahme {i+1}' if i < 3 else ''
                    
                    # Gyroscope magnitude
                    gyro_magnitude = np.sqrt(df['Gyro_X']**2 + df['Gyro_Y']**2 + df['Gyro_Z']**2)
                    ax2.plot(df['Timestamp'], gyro_magnitude, alpha=0.6, linewidth=1,
                            color=colors[i]) #, label=f'Aufnahme {i+1}' if i < 3 else ''
                except:
                    continue
            
            # Configure acceleration plot
            ax1.set_title(title, fontweight='bold')
            ax1.set_xlabel('Zeit (s)')
            ax1.set_ylabel('|Acc| (m/s²)')
            ax1.grid(True, alpha=0.3)
            if idx == 0:  # Only show legend for first subplot
                ax1.legend(loc='upper right', fontsize=8)
            
            # Configure gyroscope plot
            ax2.set_title(title, fontweight='bold')
            ax2.set_xlabel('Zeit (s)')
            ax2.set_ylabel('|Gyro| (rad/s)')
            ax2.grid(True, alpha=0.3)
            if idx == 0:  # Only show legend for first subplot
                ax2.legend(loc='upper right', fontsize=8)
    
    plt.figure(fig1.number)
    plt.tight_layout()
    comparison_path1 = os.path.join(base_folder, 'schlagarten_vergleich_beschleunigung.png')
    plt.savefig(comparison_path1, dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(fig2.number)
    plt.tight_layout()
    comparison_path2 = os.path.join(base_folder, 'schlagarten_vergleich_winkelgeschwindigkeit.png')
    plt.savefig(comparison_path2, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Vergleichsplot Beschleunigung: {comparison_path1}")
    print(f"✓ Vergleichsplot Winkelgeschwindigkeit: {comparison_path2}")

def generate_summary_statistics(base_folder):
    """Generate summary statistics for all stroke types"""
    stroke_types = {
        'vorhand_topspin': 'Vorhand Topspin',
        'vorhand_schupf': 'Vorhand Schupf',
        'rueckhand_topspin': 'Rückhand Topspin',
        'rueckhand_schupf': 'Rückhand Schupf'
    }
    
    summary_data = []
    
    for folder_name, stroke_name in stroke_types.items():
        input_folder = os.path.join(base_folder, folder_name)
        
        if os.path.exists(input_folder):
            csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
            
            max_accs = []
            max_gyros = []
            durations = []
            
            for csv_file in csv_files:
                file_path = os.path.join(input_folder, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    
                    acc_magnitude = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)
                    gyro_magnitude = np.sqrt(df['Gyro_X']**2 + df['Gyro_Y']**2 + df['Gyro_Z']**2)
                    
                    max_accs.append(acc_magnitude.max())
                    max_gyros.append(gyro_magnitude.max())
                    durations.append(df['Timestamp'].max())
                except:
                    continue
            
            if max_accs:
                summary_data.append({
                    'Schlagart': stroke_name,
                    'Anzahl Aufnahmen': len(max_accs),
                    'Ø Max Beschl. (m/s²)': np.mean(max_accs),
                    'Ø Max Winkelg. (rad/s)': np.mean(max_gyros),
                    'Ø Dauer (s)': np.mean(durations)
                })
    
    # Create summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create figure for summary table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=summary_df.round(2).values,
                        colLabels=summary_df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Zusammenfassung aller Schlagarten', fontsize=16, fontweight='bold', pad=20)
        
        summary_path = os.path.join(base_folder, 'zusammenfassung_statistiken.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Zusammenfassung: {summary_path}")

if __name__ == "__main__":
    # Process all CSV files automatically
    plot_all_csv_files()
    
    # Generate summary statistics
    base_folder = "../../rohdaten"
    print("\n📈 Erstelle Zusammenfassungsstatistiken...")
    generate_summary_statistics(base_folder)
    
    print("\n🎉 Alle Visualisierungen wurden erfolgreich erstellt!")