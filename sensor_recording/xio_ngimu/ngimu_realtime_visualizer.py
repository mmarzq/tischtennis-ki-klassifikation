"""
NGIMU Echtzeit-Visualisierung
Zeigt Live-Sensordaten vom NGIMU
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from pythonosc import dispatcher, osc_server
import threading
import time

class NGIMURealTimeVisualizer:
    def __init__(self, ip="0.0.0.0", port=9000, window_size=500):
        self.ip = ip
        self.port = port
        self.window_size = window_size
        
        # Daten-Buffer
        self.time_buffer = deque(maxlen=window_size)
        self.acc_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        self.gyro_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        
        # Bewegungsintensität
        self.movement_intensity = deque(maxlen=window_size)
        
        # OSC Setup
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/sensors", self.handle_sensors)
        
        self.server = None
        self.server_thread = None
        self.start_time = time.time()
        
        # Plot Setup
        self.fig = None
        self.axes = None
        self.lines = {}
        
    def handle_sensors(self, address, *args):
        """Verarbeitet eingehende Sensordaten"""
        if len(args) >= 10:
            current_time = time.time() - self.start_time
            self.time_buffer.append(current_time)
            
            # Gyroskop
            self.gyro_buffers['x'].append(args[1])
            self.gyro_buffers['y'].append(args[2])
            self.gyro_buffers['z'].append(args[3])
            
            # Beschleunigung
            self.acc_buffers['x'].append(args[4])
            self.acc_buffers['y'].append(args[5])
            self.acc_buffers['z'].append(args[6])
            
            # Bewegungsintensität berechnen
            acc_mag = np.sqrt(args[4]**2 + args[5]**2 + args[6]**2)
            gyro_mag = np.sqrt(args[1]**2 + args[2]**2 + args[3]**2)
            intensity = acc_mag + (gyro_mag / 100)
            self.movement_intensity.append(intensity)
    
    def start_server(self):
        """Startet den OSC-Server"""
        self.server = osc_server.ThreadingOSCUDPServer(
            (self.ip, self.port), self.dispatcher
        )
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"OSC-Server läuft auf {self.ip}:{self.port}")
    
    def stop_server(self):
        """Stoppt den OSC-Server"""
        if self.server:
            self.server.shutdown()
    
    def setup_plot(self):
        """Initialisiert die Plots"""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('NGIMU Echtzeit-Daten', fontsize=16)
        
        # Beschleunigung Plot
        self.axes[0].set_ylabel('Beschleunigung (g)')
        self.axes[0].set_ylim(-4, 4)
        self.axes[0].grid(True, alpha=0.3)
        self.lines['acc_x'], = self.axes[0].plot([], [], 'r-', label='X', linewidth=2)
        self.lines['acc_y'], = self.axes[0].plot([], [], 'g-', label='Y', linewidth=2)
        self.lines['acc_z'], = self.axes[0].plot([], [], 'b-', label='Z', linewidth=2)
        self.axes[0].legend(loc='upper right')
        
        # Gyroskop Plot
        self.axes[1].set_ylabel('Winkelgeschw. (°/s)')
        self.axes[1].set_ylim(-500, 500)
        self.axes[1].grid(True, alpha=0.3)
        self.lines['gyro_x'], = self.axes[1].plot([], [], 'r-', label='X', linewidth=2)
        self.lines['gyro_y'], = self.axes[1].plot([], [], 'g-', label='Y', linewidth=2)
        self.lines['gyro_z'], = self.axes[1].plot([], [], 'b-', label='Z', linewidth=2)
        self.axes[1].legend(loc='upper right')
        
        # Bewegungsintensität Plot
        self.axes[2].set_ylabel('Bewegungsintensität')
        self.axes[2].set_xlabel('Zeit (s)')
        self.axes[2].set_ylim(0, 10)
        self.axes[2].grid(True, alpha=0.3)
        self.lines['intensity'], = self.axes[2].plot([], [], 'y-', linewidth=3)
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        """Aktualisiert die Plots"""
        if len(self.time_buffer) > 1:
            time_data = list(self.time_buffer)
            
            # X-Achsen-Limits anpassen
            for ax in self.axes:
                ax.set_xlim(max(0, time_data[-1] - 5), time_data[-1] + 0.5)
            
            # Beschleunigung aktualisieren
            self.lines['acc_x'].set_data(time_data, list(self.acc_buffers['x']))
            self.lines['acc_y'].set_data(time_data, list(self.acc_buffers['y']))
            self.lines['acc_z'].set_data(time_data, list(self.acc_buffers['z']))
            
            # Gyroskop aktualisieren
            self.lines['gyro_x'].set_data(time_data, list(self.gyro_buffers['x']))
            self.lines['gyro_y'].set_data(time_data, list(self.gyro_buffers['y']))
            self.lines['gyro_z'].set_data(time_data, list(self.gyro_buffers['z']))
            
            # Bewegungsintensität aktualisieren
            self.lines['intensity'].set_data(time_data, list(self.movement_intensity))
            
            # Schlag-Erkennung (visuell)
            if self.movement_intensity and self.movement_intensity[-1] > 3:
                self.axes[2].set_facecolor((0.2, 0, 0, 0.3))  # Rötlicher Hintergrund
            else:
                self.axes[2].set_facecolor((0, 0, 0, 0))
        
        return list(self.lines.values())
    
    def run(self):
        """Startet die Visualisierung"""
        print("=== NGIMU Echtzeit-Visualisierung ===")
        print("Warte auf Daten vom NGIMU...")
        print("Drücken Sie Ctrl+C zum Beenden")
        
        self.start_server()
        self.setup_plot()
        
        # Animation starten
        anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nBeende Visualisierung...")
        finally:
            self.stop_server()

def main():
    """Hauptfunktion"""
    visualizer = NGIMURealTimeVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
