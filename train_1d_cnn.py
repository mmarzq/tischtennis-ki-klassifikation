"""
1D CNN Modell für Tischtennisschlag-Klassifikation
Training und Evaluation des neuronalen Netzes
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

class TischtennisCNN:
    def __init__(self, input_shape, num_classes=6):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Erstellt die 1D CNN Architektur - angepasst für 10 Features"""
        self.model = models.Sequential([
            # Input Layer
            layers.Input(shape=self.input_shape),
            
            # 1. Convolutional Block
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # 2. Convolutional Block
            layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            # 3. Convolutional Block
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.3),
            
            # Dense Layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Modell kompilieren
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Trainiert das Modell"""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test, class_names):
        """Evaluiert das Modell"""
        # Vorhersagen
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Metriken berechnen
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self.plot_confusion_matrix(cm, class_names)
        
        return test_accuracy, y_pred_classes
    
    def plot_confusion_matrix(self, cm, class_names):
        """Zeigt die Confusion Matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Wahre Klasse')
        plt.xlabel('Vorhergesagte Klasse')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png')
        plt.show()
    
    def plot_training_history(self):
        """Zeigt den Trainingsverlauf"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/training_history.png')
        plt.show()

def load_and_prepare_data():
    """Lädt und bereitet die Daten vor"""
    # Daten laden
    X = np.load('processed_data/X_data.npy')
    y = np.load('processed_data/y_labels.npy')
    
    # Scaler laden
    with open('processed_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Feature-Namen laden (falls vorhanden)
    try:
        with open('processed_data/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
            print(f"Geladene Features ({len(feature_names)}): {feature_names}")
    except:
        print("Feature-Namen nicht gefunden")
    
    # Normalisierung
    n_samples, n_timesteps, n_features = X.shape
    print(f"Datenform: {n_samples} Samples, {n_timesteps} Zeitschritte, {n_features} Features")
    
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_flat)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"\nDatensätze:")
    print(f"Training: {X_train.shape[0]} Samples")
    print(f"Validation: {X_val.shape[0]} Samples")
    print(f"Test: {X_test.shape[0]} Samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Haupttraining"""
    # Klassen definieren
    class_names = ['Vorhand Topspin', 'Vorhand Schupf', 
                   'Rückhand Topspin', 'Rückhand Schupf']
    
    # Ordner erstellen
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    # Modell erstellen
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    cnn = TischtennisCNN(input_shape, num_classes=4)
    
    # Modell bauen
    model = cnn.build_model()
    print("\nModell-Architektur:")
    model.summary()
    
    # Training
    print("\nStarte Training...")
    history = cnn.train(X_train, y_train, X_val, y_val, 
                       epochs=100, batch_size=32)
    
    # Trainingsverlauf anzeigen
    cnn.plot_training_history()
    
    # Evaluation
    print("\nEvaluation auf Testdaten:")
    test_acc, predictions = cnn.evaluate(X_test, y_test, class_names)
    
    # Modell speichern
    model.save('models/final_tischtennis_model.h5')
    
    # Modell-Informationen speichern
    model_info = {
        'input_shape': input_shape,
        'num_classes': 4,
        'num_features': X_train.shape[2],
        'class_names': class_names,
        'test_accuracy': test_acc
    }
    
    # Feature-Namen hinzufügen falls vorhanden
    try:
        with open('processed_data/feature_names.pkl', 'rb') as f:
            model_info['feature_names'] = pickle.load(f)
    except:
        pass
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print("\nTraining abgeschlossen!")
    print(f"Finale Test-Genauigkeit: {test_acc:.4f}")
    print(f"Modell verwendet {X_train.shape[2]} Features")

if __name__ == "__main__":
    main()
