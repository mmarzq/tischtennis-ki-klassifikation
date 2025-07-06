"""
NGIMU Echtzeit-Visualisierung
Zeigt Live-Sensordaten vom NGIMU mit der gleichen Empfangsmethode wie xio_ngimu_recorder
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import socket
import time
import osc_decoder
import threading

class NGIMURealTimeVisualizer:
    def __init__(self, window_size=500):
        self.window_size = window_size
        
        # Netzwerk-Konfiguration
        self.NGIMU_IP = "192.168.1.1"
        self.NGIMU_SEND_PORT = 9000
        self.RECEIVE_PORTS = [8001, 8010, 8011, 8012]
        
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
        self.lin_acc_buffers = {
            'x': deque(maxlen=window_size),
            'y': deque(maxlen=window_size),
            'z': deque(maxlen=window_size)
        }
        
        # Bewegungsintensität
        self.movement_intensity = deque(maxlen=window_size)
        
        # Socket-Verbindungen
        self.send_socket = None
        self.receive_sockets = []
        self.running = False
        self.data_thread = None
        
        self.start_time = time.time()
        
        # Plot Setup
        self.fig = None
        self.axes = None
        self.lines = {}
        
    def setup_sockets(self):
        """Erstellt und konfiguriert die Socket-Verbindungen"""
        # Socket für Senden erstellen
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Sockets für Empfang erstellen
        self.receive_sockets = []
        for port in self.RECEIVE_PORTS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("", port))
            sock.setblocking(False)
            self.receive_sockets.append(sock)
        
        # Send /identify message an NGIMU
        self.send_socket.sendto(bytes("/identify\0\0\0,\0\0\0", "utf-8"), 
                                (self.NGIMU_IP, self.NGIMU_SEND_PORT))
        
        print(f"Socket-Verbindungen erstellt auf Ports: {self.RECEIVE_PORTS}")
    
    def receive_data(self):
        """Empfängt Daten vom NGIMU (läuft in separatem Thread)"""
        while self.running:
            # Daten von allen Sockets empfangen
            for udp_socket in self.receive_sockets:
                try:
                    data, addr = udp_socket.recvfrom(2048)
                except socket.error:
                    pass
                else:
                    # OSC Messages dekodieren
                    for message in osc_decoder.decode(data):
                        if len(message) >= 2:
                            current_time = time.time() - self.start_time
                            osc_address = message[1]
                            
                            # Sensordaten verarbeiten
                            if osc_address == '/sensors' and len(message) == 12:
                                self.time_buffer.append(current_time)
                                
                                # Gyroskop (Index 2-4)
                                self.gyro_buffers['x'].append(message[2])
                                self.gyro_buffers['y'].append(message[3])
                                self.gyro_buffers['z'].append(message[4])
                                
                                # Beschleunigung (Index 5-7)
                                self.acc_buffers['x'].append(message[5])
                                self.acc_buffers['y'].append(message[6])
                                self.acc_buffers['z'].append(message[7])
                                
                                # Bewegungsintensität berechnen
                                acc_mag = np.sqrt(message[5]**2 + message[6]**2 + message[7]**2)
                                gyro_mag = np.sqrt(message[2]**2 + message[3]**2 + message[4]**2)
                                intensity = acc_mag + (gyro_mag / 100)
                                self.movement_intensity.append(intensity)
                            
                            # Lineare Beschleunigung verarbeiten
                            elif osc_address == '/linear' and len(message) == 5:
                                # Sicherstellen dass wir synchron mit den Sensordaten sind
                                if len(self.time_buffer) > 0:
                                    self.lin_acc_buffers['x'].append(message[2])
                                    self.lin_acc_buffers['y'].append(message[3])
                                    self.lin_acc_buffers['z'].append(message[4])
            
            # Kurze Pause um CPU zu schonen
            time.sleep(0.001)
    
    def start_data_reception(self):
        """Startet den Datenempfang in einem separaten Thread"""
        self.running = True
        self.data_thread = threading.Thread(target=self.receive_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        print("Datenempfang gestartet...")
    
    def stop_data_reception(self):
        """Stoppt den Datenempfang"""
        self.running = False
        if self.data_thread:
            self.data_thread.join()
        
        # Sockets schließen
        for sock in self.receive_sockets:
            sock.close()
        if self.send_socket:
            self.send_socket.close()
    
    def setup_plot(self):
        """Initialisiert die Plots"""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 12))
        self.fig.suptitle('NGIMU Echtzeit-Daten', fontsize=16)
        
        # Beschleunigung Plot
        self.axes[0].set_ylabel('Beschleunigung (g)')
        self.axes[0].set_ylim(-4, 4)
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_title('Rohe Beschleunigung', fontsize=12)
        self.lines['acc_x'], = self.axes[0].plot([], [], 'r-', label='X', linewidth=2)
        self.lines['acc_y'], = self.axes[0].plot([], [], 'g-', label='Y', linewidth=2)
        self.lines['acc_z'], = self.axes[0].plot([], [], 'b-', label='Z', linewidth=2)
        self.axes[0].legend(loc='upper right')
        
        # Gyroskop Plot
        self.axes[1].set_ylabel('Winkelgeschw. (°/s)')
        self.axes[1].set_ylim(-500, 500)
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].set_title('Gyroskop', fontsize=12)
        self.lines['gyro_x'], = self.axes[1].plot([], [], 'r-', label='X', linewidth=2)
        self.lines['gyro_y'], = self.axes[1].plot([], [], 'g-', label='Y', linewidth=2)
        self.lines['gyro_z'], = self.axes[1].plot([], [], 'b-', label='Z', linewidth=2)
        self.axes[1].legend(loc='upper right')
        
        # Lineare Beschleunigung Plot
        self.axes[2].set_ylabel('Lin. Beschl. (g)')
        self.axes[2].set_ylim(-4, 4)
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].set_title('Lineare Beschleunigung (ohne Gravitation)', fontsize=12)
        self.lines['lin_acc_x'], = self.axes[2].plot([], [], 'r-', label='X', linewidth=2)
        self.lines['lin_acc_y'], = self.axes[2].plot([], [], 'g-', label='Y', linewidth=2)
        self.lines['lin_acc_z'], = self.axes[2].plot([], [], 'b-', label='Z', linewidth=2)
        self.axes[2].legend(loc='upper right')
        
        # Bewegungsintensität Plot
        self.axes[3].set_ylabel('Bewegungsintensität')
        self.axes[3].set_xlabel('Zeit (s)')
        self.axes[3].set_ylim(0, 10)
        self.axes[3].grid(True, alpha=0.3)
        self.axes[3].set_title('Gesamte Bewegungsintensität', fontsize=12)
        self.lines['intensity'], = self.axes[3].plot([], [], 'y-', linewidth=3)
        
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
            
            # Lineare Beschleunigung aktualisieren
            if len(self.lin_acc_buffers['x']) > 0:
                # Zeitdaten für lineare Beschleunigung anpassen (falls weniger Datenpunkte)
                lin_time_data = time_data[-len(self.lin_acc_buffers['x']):]
                self.lines['lin_acc_x'].set_data(lin_time_data, list(self.lin_acc_buffers['x']))
                self.lines['lin_acc_y'].set_data(lin_time_data, list(self.lin_acc_buffers['y']))
                self.lines['lin_acc_z'].set_data(lin_time_data, list(self.lin_acc_buffers['z']))
            
            # Bewegungsintensität aktualisieren
            self.lines['intensity'].set_data(time_data, list(self.movement_intensity))
            
            # Schlag-Erkennung (visuell)
            if self.movement_intensity and self.movement_intensity[-1] > 3:
                self.axes[3].set_facecolor((0.2, 0, 0, 0.3))  # Rötlicher Hintergrund
            else:
                self.axes[3].set_facecolor((0, 0, 0, 0))
        
        return list(self.lines.values())
    
    def run(self):
        """Startet die Visualisierung"""
        print("=== NGIMU Echtzeit-Visualisierung ===")
        print(f"Verbinde mit NGIMU auf {self.NGIMU_IP}...")
        print("Warte auf Daten...")
        print("Drücken Sie Ctrl+C zum Beenden")
        
        self.setup_sockets()
        self.start_data_reception()
        self.setup_plot()
        
        # Animation starten
        anim = FuncAnimation(self.fig, self.update_plot, interval=50, blit=True)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nBeende Visualisierung...")
        finally:
            self.stop_data_reception()

def main():
    """Hauptfunktion"""
    visualizer = NGIMURealTimeVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
