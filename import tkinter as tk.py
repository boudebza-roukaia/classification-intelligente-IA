import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class DiabetesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🩺 Prédiction du Diabète - CDC Dataset")
        self.root.geometry("900x600")
        
        # Dataset
        self.data = None
        self.model = None
        
        # Boutons principaux
        tk.Button(root, text="📂 Charger Dataset", command=self.load_data, width=25).pack(pady=10)
        tk.Button(root, text="⚙️ Entraîner Modèle (Random Forest)", command=self.train_model, width=25).pack(pady=10)
        tk.Button(root, text="📊 Évaluer", command=self.evaluate_model, width=25).pack(pady=10)
        
        # Zone d'affichage
        self.text_area = tk.Text(root, height=20, width=100)
        self.text_area.pack(pady=10)
    
    def load_data(self):
        path = filedialog.askopenfilename(title="Choisir le fichier CSV")
        if path:
            self.data = pd.read_csv(path)
            self.text_area.insert(tk.END, f"✅ Dataset chargé : {path}\nNombre d'échantillons : {len(self.data)}\n\n")

    def train_model(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Veuillez d'abord charger le dataset.")
            return
        
        X = self.data.drop(columns=['Diabetes_012'])
        y = self.data['Diabetes_012']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        self.text_area.insert(tk.END, "✅ Modèle entraîné avec succès (Random Forest)\n\n")

    def evaluate_model(self):
        if self.model is None:
            messagebox.showerror("Erreur", "Veuillez d'abord entraîner le modèle.")
            return
        
        X = self.data.drop(columns=['Diabetes_012'])
        y = self.data['Diabetes_012']
        y_pred = self.model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        self.text_area.insert(tk.END, f"📊 Évaluation du modèle :\nAccuracy = {acc:.2f}\n")
        self.text_area.insert(tk.END, f"Rapport de classification :\n{report}\n")
        self.text_area.insert(tk.END, f"Matrice de confusion :\n{cm}\n\n")

root = tk.Tk()
app = DiabetesApp(root)
root.mainloop()
