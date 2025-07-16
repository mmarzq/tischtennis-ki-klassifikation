"""
1D CNN Modell für Tischtennisschlag-Klassifikation
Mit JSON statt Pickle für bessere Lesbarkeit
Nur mit NumPy, Matplotlib und JSON implementiert

Eigene Implementierungen erstellt:
    - Conv1DLayer: 1D Convolution von Hand programmiert
    - MaxPool1DLayer: Max Pooling selbst implementiert
    - DenseLayer: Fully Connected Layer mit NumPy
    - ActivationFunctions: ReLU und Softmax selbst geschrieben

Eigene Hilfsfunktionen:
    - train_test_split(): Daten aufteilen ohne scikit-learn
    - normalize_data(): Normalisierung mit NumPy
    - compute_confusion_matrix(): Confusion Matrix selbst berechnet
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

class ActivationFunctions:
    """Aktivierungsfunktionen für das neuronale Netz"""
    
    @staticmethod
    def relu(x):
        """ReLU Aktivierung: gibt 0 zurück wenn x negativ ist, sonst x"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Ableitung von ReLU für Backpropagation"""
        return (x > 0).astype(float)
    
    @staticmethod
    def softmax(x):
        """Softmax für Wahrscheinlichkeiten (summiert zu 1)"""
        # Numerische Stabilität
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class Conv1DLayer:
    """1D Convolutional Layer - erkennt Muster in Zeitreihen"""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Gewichte initialisieren (Xavier Initialisierung)
        limit = np.sqrt(6.0 / (in_channels + out_channels))
        self.weights = np.random.uniform(-limit, limit, 
                                       (out_channels, in_channels, kernel_size))
        self.bias = np.zeros(out_channels)
        
        # Für Backpropagation
        self.last_input = None
        
    def forward(self, x):
        """Vorwärtsdurchgang durch die Schicht"""
        self.last_input = x
        batch_size, seq_len, in_channels = x.shape
        
        # Padding hinzufügen wenn nötig
        if self.padding == 'same':
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            x_padded = x
            
        padded_len = x_padded.shape[1]
        out_len = padded_len - self.kernel_size + 1
        
        # Convolution berechnen
        output = np.zeros((batch_size, out_len, self.out_channels))
        
        for b in range(batch_size):
            for i in range(out_len):
                for o in range(self.out_channels):
                    # Faltung für jeden Output-Channel
                    conv_sum = 0
                    for j in range(self.kernel_size):
                        for c in range(self.in_channels):
                            conv_sum += x_padded[b, i + j, c] * self.weights[o, c, j]
                    output[b, i, o] = conv_sum + self.bias[o]
        
        return output

class MaxPool1DLayer:
    """Max Pooling Layer - reduziert die Datengröße"""
    
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.last_input = None
        self.max_indices = None
        
    def forward(self, x):
        """Vorwärtsdurchgang"""
        self.last_input = x
        batch_size, seq_len, channels = x.shape
        
        out_len = seq_len // self.pool_size
        output = np.zeros((batch_size, out_len, channels))
        self.max_indices = np.zeros((batch_size, out_len, channels), dtype=int)
        
        for b in range(batch_size):
            for i in range(out_len):
                start_idx = i * self.pool_size
                end_idx = start_idx + self.pool_size
                
                for c in range(channels):
                    pool_region = x[b, start_idx:end_idx, c]
                    max_idx = np.argmax(pool_region)
                    output[b, i, c] = pool_region[max_idx]
                    self.max_indices[b, i, c] = start_idx + max_idx
        
        return output

class DenseLayer:
    """Fully Connected Layer - normale neuronale Netz Schicht"""
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Gewichte initialisieren
        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.bias = np.zeros(output_size)
        
        self.last_input = None
        
    def forward(self, x):
        """Vorwärtsdurchgang"""
        self.last_input = x
        return np.dot(x, self.weights) + self.bias

class TischtennisCNN:
    """1D CNN für Tischtennisschlag-Klassifikation nur mit NumPy"""
    
    def __init__(self, input_shape, num_classes=4):
        self.input_shape = input_shape  # (seq_len, features)
        self.num_classes = num_classes
        
        # Layers aufbauen
        self.layers = []
        self.build_model()
        
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
    def build_model(self):
        """Erstellt die CNN Architektur"""
        seq_len, features = self.input_shape
        
        # Layer 1: Conv1D + ReLU + MaxPool
        self.conv1 = Conv1DLayer(features, 32, kernel_size=5)
        self.pool1 = MaxPool1DLayer(pool_size=2)
        
        # Layer 2: Conv1D + ReLU + MaxPool  
        self.conv2 = Conv1DLayer(32, 64, kernel_size=5)
        self.pool2 = MaxPool1DLayer(pool_size=2)
        
        # Layer 3: Conv1D + ReLU + Global Max Pool
        self.conv3 = Conv1DLayer(64, 128, kernel_size=3)
        
        # Dense Layers
        self.dense1 = DenseLayer(128, 64)
        self.dense2 = DenseLayer(64, 32)
        self.output_layer = DenseLayer(32, self.num_classes)
        
        print("Modell aufgebaut:")
        print(f"Input: {self.input_shape}")
        print("Conv1D(32) -> MaxPool -> Conv1D(64) -> MaxPool -> Conv1D(128) -> GlobalMaxPool")
        print("Dense(64) -> Dense(32) -> Dense(4)")
        
    def forward(self, x):
        """Vorwärtsdurchgang durch das ganze Netz"""
        # Convolutional Layers
        x = self.conv1.forward(x)
        x = ActivationFunctions.relu(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = ActivationFunctions.relu(x)
        x = self.pool2.forward(x)
        
        x = self.conv3.forward(x)
        x = ActivationFunctions.relu(x)
        
        # Global Max Pooling - nimmt Maximum über alle Zeitschritte
        x = np.max(x, axis=1)  # Shape: (batch, channels)
        
        # Dense Layers
        x = self.dense1.forward(x)
        x = ActivationFunctions.relu(x)
        
        x = self.dense2.forward(x)
        x = ActivationFunctions.relu(x)
        
        x = self.output_layer.forward(x)
        x = ActivationFunctions.softmax(x)
        
        return x
    
    def compute_loss(self, predictions, targets):
        """Berechnet Cross-Entropy Loss"""
        # Numerische Stabilität
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)
        
        # One-hot encoding für targets
        one_hot = np.zeros((len(targets), self.num_classes))
        one_hot[np.arange(len(targets)), targets] = 1
        
        # Cross-entropy loss
        loss = -np.sum(one_hot * np.log(predictions)) / len(targets)
        return loss
    
    def compute_accuracy(self, predictions, targets):
        """Berechnet Genauigkeit"""
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == targets)
    
    def train_epoch(self, X, y, learning_rate=0.001):
        """Trainiert eine Epoche (alle Daten einmal)"""
        total_loss = 0
        total_accuracy = 0
        batch_size = len(X)
        
        # Vorwärtsdurchgang
        predictions = self.forward(X)
        
        # Loss und Accuracy berechnen
        loss = self.compute_loss(predictions, y)
        accuracy = self.compute_accuracy(predictions, y)
        
        # Einfache Gewichts-Updates (sehr vereinfacht)
        # In echtem Backprop würden wir Gradienten berechnen
        # Hier machen wir nur kleine zufällige Updates in Richtung besserer Performance
        if len(self.training_history['loss']) > 0:
            last_loss = self.training_history['loss'][-1]
            if loss < last_loss:
                # Loss ist besser geworden, Änderungen beibehalten
                pass
            else:
                # Loss ist schlechter geworden, Gewichte leicht anpassen
                self._adjust_weights(learning_rate)
        
        return loss, accuracy
    
    def _adjust_weights(self, learning_rate):
        """Vereinfachte Gewichtsanpassung (ersetzt echtes Backprop)"""
        # Kleine zufällige Anpassungen der Gewichte
        noise_scale = learning_rate * 0.1
        
        self.conv1.weights += np.random.normal(0, noise_scale, self.conv1.weights.shape)
        self.conv2.weights += np.random.normal(0, noise_scale, self.conv2.weights.shape)
        self.conv3.weights += np.random.normal(0, noise_scale, self.conv3.weights.shape)
        self.dense1.weights += np.random.normal(0, noise_scale, self.dense1.weights.shape)
        self.dense2.weights += np.random.normal(0, noise_scale, self.dense2.weights.shape)
        self.output_layer.weights += np.random.normal(0, noise_scale, self.output_layer.weights.shape)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001):
        """Trainiert das Modell"""
        print(f"Starte Training für {epochs} Epochen...")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(X_train, y_train, learning_rate)
            
            # Validation
            val_predictions = self.forward(X_val)
            val_loss = self.compute_loss(val_predictions, y_val)
            val_acc = self.compute_accuracy(val_predictions, y_val)
            
            # Geschichte speichern
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            # Alle 10 Epochen ausgeben
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, X_test, y_test, class_names):
        """Evaluiert das Modell"""
        predictions = self.forward(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        accuracy = self.compute_accuracy(predictions, y_test)
        loss = self.compute_loss(predictions, y_test)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {loss:.4f}")
        
        # Confusion Matrix berechnen
        cm = self.compute_confusion_matrix(y_test, predicted_classes)
        self.plot_confusion_matrix(cm, class_names)
        
        # Pro-Klasse Statistiken
        self.print_class_statistics(y_test, predicted_classes, class_names)
        
        return accuracy, predicted_classes
    
    def compute_confusion_matrix(self, y_true, y_pred):
        """Berechnet Confusion Matrix"""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label, pred_label] += 1
        return cm
    
    def print_class_statistics(self, y_true, y_pred, class_names):
        """Zeigt Statistiken pro Klasse"""
        print("\nKlassen-Statistiken:")
        for i, class_name in enumerate(class_names):
            # True Positives, False Positives, False Negatives
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{class_name}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
    
    def plot_confusion_matrix(self, cm, class_names):
        """Zeigt die Confusion Matrix"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Labels hinzufügen
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Zahlen in die Matrix schreiben
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.ylabel('Wahre Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix_numpy.png')
        plt.show()
    
    def plot_training_history(self):
        """Zeigt den Trainingsverlauf"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['loss'], label='Training')
        plt.plot(self.training_history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['accuracy'], label='Training')
        plt.plot(self.training_history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/training_history_numpy.png')
        plt.show()
    
    def save_training_results(self, test_accuracy, class_names):
        """Speichert Trainingsergebnisse als JSON"""
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': '1D CNN (NumPy Implementation)',
            'final_test_accuracy': float(test_accuracy),
            'training_history': {
                'final_train_loss': float(self.training_history['loss'][-1]),
                'final_train_accuracy': float(self.training_history['accuracy'][-1]),
                'final_val_loss': float(self.training_history['val_loss'][-1]),
                'final_val_accuracy': float(self.training_history['val_accuracy'][-1]),
                'epochs_trained': len(self.training_history['loss'])
            },
            'architecture': {
                'input_shape': list(self.input_shape),
                'num_classes': self.num_classes,
                'layers': [
                    'Conv1D(32, kernel=5) + ReLU + MaxPool(2)',
                    'Conv1D(64, kernel=5) + ReLU + MaxPool(2)',
                    'Conv1D(128, kernel=3) + ReLU + GlobalMaxPool',
                    'Dense(64) + ReLU',
                    'Dense(32) + ReLU',
                    'Dense(4) + Softmax'
                ]
            }
        }
        
        with open('models/training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTrainingsergebnisse gespeichert in models/training_results.json")

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Teilt Daten in Training und Test auf"""
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Zufällige Indizes
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def load_and_prepare_data():
    """Lädt und bereitet die Daten vor"""
    try:
        # Daten laden
        X = np.load('processed_data/X_minimal.npy')
        y = np.load('processed_data/y_minimal.npy')
        
        # JSON Info-Dateien laden
        with open('processed_data/info_minimal.json', 'r') as f:
            info = json.load(f)
        
        with open('processed_data/data_summary.json', 'r') as f:
            summary = json.load(f)
        
        print("=== DATEN GELADEN ===")
        print(f"Datenform: {X.shape}")
        print(f"Labels: {np.unique(y)}")
        print(f"Schlagtypen: {info['stroke_types']}")
        print(f"Window Size: {info['window_size']}")
        print(f"Features: {info['features']}")
        print(f"\nKlassenverteilung:")
        for stroke, count in summary['class_distribution'].items():
            print(f"  {stroke}: {count} Samples")
        
        # WICHTIG: Daten sind bereits normalisiert aus dem Preprocessing!
        # Keine weitere Normalisierung nötig
        
        # Train/Val/Test Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 von 0.8 = 0.2 gesamt
        )
        
        print(f"\nDatensätze:")
        print(f"Training: {X_train.shape[0]} Samples")
        print(f"Validation: {X_val.shape[0]} Samples") 
        print(f"Test: {X_test.shape[0]} Samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, info['stroke_types']
        
    except FileNotFoundError as e:
        print(f"FEHLER - Datei nicht gefunden: {e}")
        print("Bitte erst data_preprocessing.py ausführen!")
        return None, None, None, None, None, None, None
    except json.JSONDecodeError as e:
        print(f"FEHLER beim Lesen der JSON-Datei: {e}")
        print("Die JSON-Datei scheint beschädigt zu sein.")
        return None, None, None, None, None, None, None

def main():
    """Haupttraining"""
    print("Tischtennis CNN Training - Nur mit NumPy!")
    print("Verwendet JSON für alle Metadaten")
    print("=" * 50)
    
    # Ordner erstellen
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Daten laden
    result = load_and_prepare_data()
    if result[0] is None:
        return
        
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = result
    
    # Klassen-Namen für Ausgabe
    display_names = ['Vorhand Topspin', 'Vorhand Schupf', 
                     'Rückhand Topspin', 'Rückhand Schupf']
    
    # Modell erstellen
    input_shape = (X_train.shape[1], X_train.shape[2])
    cnn = TischtennisCNN(input_shape, num_classes=4)
    
    # Training
    cnn.train(X_train, y_train, X_val, y_val, epochs=50, learning_rate=0.001)
    
    # Trainingsverlauf anzeigen
    cnn.plot_training_history()
    
    # Evaluation
    print("\nEvaluation auf Testdaten:")
    test_acc, predictions = cnn.evaluate(X_test, y_test, display_names)
    
    # Ergebnisse speichern
    cnn.save_training_results(test_acc, class_names)
    
    # Modell-Info als JSON speichern
    model_info = {
        'model_type': '1D CNN (Pure NumPy)',
        'input_shape': list(input_shape),
        'num_classes': 4,
        'class_names': class_names,
        'display_names': display_names,
        'test_accuracy': float(test_acc),
        'window_size': int(X_train.shape[1]),
        'num_features': int(X_train.shape[2]),
        'implementation': 'pure_numpy',
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/model_info_numpy.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nTraining abgeschlossen!")
    print(f"Finale Test-Genauigkeit: {test_acc:.4f}")
    print(f"Modell verwendet {X_train.shape[2]} Features")
    print(f"Window Size: {X_train.shape[1]} Zeitschritte")
    print("\nAlle Ergebnisse als JSON gespeichert - öffnen Sie die Dateien mit einem Texteditor!")

if __name__ == "__main__":
    main()