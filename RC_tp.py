from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from scipy.stats import mode
from sklearn.decomposition import PCA
from tkinter import ttk
from PIL import Image, ImageTk

class AppStyle:
    BACKGROUND = "#2E3440"
    FOREGROUND = "#D8DEE9"
    ACCENT = "#88C0D0"
    SECONDARY = "#5E81AC"
    HIGHLIGHT = "#BF616A"
    BUTTON_BG = "#434C5E"
    TEXT_BG = "#3B4252"
    ENTRY_BG = "#4C566A"

def create_welcome_screen(root):
    style = ttk.Style()
    style.theme_use('clam')
    
    style.configure('.', background=AppStyle.BACKGROUND)
    style.configure('TFrame', background=AppStyle.BACKGROUND)
    style.configure('TLabel', 
                   background=AppStyle.BACKGROUND, 
                   foreground=AppStyle.FOREGROUND,
                   font=("Arial", 12))
    style.configure('Title.TLabel', 
                   font=("Arial", 18, "bold"),
                   foreground=AppStyle.ACCENT)
    style.configure('TButton', 
                   background=AppStyle.BUTTON_BG,
                   foreground=AppStyle.FOREGROUND,
                   borderwidth=2,
                   relief="raised",
                   font=("Arial", 10, "bold"))
    style.map('TButton',
             background=[('active', AppStyle.SECONDARY), ('pressed', AppStyle.HIGHLIGHT)])

    welcome_frame = ttk.Frame(root, style='TFrame')
    welcome_frame.pack(fill=tk.BOTH, expand=True)

    title_text = "Développement d'une interface intelligente pour l'analyse automatique\net la classification supervisée de données\nà l'aide de différents algorithmes d'apprentissage automatique"
    title_label = ttk.Label(
        welcome_frame,
        text=title_text,
        style='Title.TLabel',
        justify='center'
    )
    title_label.pack(pady=40)

    students_label = ttk.Label(
        welcome_frame,
        text="Réalisé par : Roukaia Boudebza et Hadjoub Dhekra",
        style='TLabel'
    )
    students_label.pack(pady=10)

    professors_label = ttk.Label(
        welcome_frame,
        text="Encadré par : Dr. Djebbar",
        style='TLabel'
    )
    professors_label.pack(pady=10)

    row_frame = ttk.Frame(welcome_frame, style='TFrame')
    row_frame.pack(pady=20)

    image = Image.open("iati.png")
    image = image.resize((420, 420), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)

    image_label = tk.Label(row_frame, image=photo, bg=AppStyle.BACKGROUND)
    image_label.image = photo
    image_label.pack(side=tk.RIGHT, padx=20)

    buttons_frame = ttk.Frame(row_frame, style='TFrame')
    buttons_frame.pack(side=tk.RIGHT, padx=20, pady=10)
    enter_btn = ttk.Button(
        buttons_frame,
        text="Entrer dans l'interface de classification",
        style='TButton',
        command=lambda: launch_classification_interface(welcome_frame)
    )
    enter_btn.pack(pady=10)

def launch_classification_interface(welcome_frame):
    welcome_frame.destroy()
    ClassificationApp(root)

class ClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification Tool")
        self.root.geometry("1200x500")
        self.root.configure(bg=AppStyle.BACKGROUND)
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background=AppStyle.BACKGROUND)
        self.style.configure('TCombobox', 
                           fieldbackground=AppStyle.ENTRY_BG,
                           background=AppStyle.BACKGROUND,
                           foreground=AppStyle.FOREGROUND)
        
        self.data = None
        self.raw_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {}
        self.results = {}
        self.best_model = None
        
        self.create_widgets()
        plt.style.use('dark_background')

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        button_frame = ttk.Frame(main_frame, style='TFrame')
        button_frame.pack(fill=tk.X)
        
        buttons = [
            ("Nouvelle Base", self.confirm_reset),
            ("Charger Dataset", self.load_dataset),
            ("Analyse ACP", self.acp_analysis),
            ("Prétraitement", self.preprocess),
            ("Vérifier Équilibre", self.check_balance),
            ("Split Train/Test", self.split_data),
            ("Exécuter Modèles", self.run_models),
            ("Meilleur Modèle", self.select_best_model),
            ("Évaluer Modèle", self.evaluate_model)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(button_frame, 
                            text=text, 
                            command=command,
                            style='TButton')
            btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        
        model_frame = ttk.Frame(main_frame, style='TFrame')
        model_frame.pack(fill=tk.X, pady=10)
        
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(model_frame, 
                                         textvariable=self.model_var,
                                         style='TCombobox')
        self.model_combobox.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.results_text = tk.Text(main_frame, 
                                   wrap=tk.WORD, 
                                   height=10,
                                   bg=AppStyle.TEXT_BG,
                                   fg=AppStyle.FOREGROUND,
                                   insertbackground=AppStyle.FOREGROUND,
                                   font=("Consolas", 12))
        scrollbar = ttk.Scrollbar(main_frame, 
                                 orient=tk.VERTICAL, 
                                 command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.figure_frame = ttk.Frame(main_frame, style='TFrame')
        self.figure_frame.pack(fill=tk.BOTH, expand=True)

    def reset_interface(self):
        """Réinitialise complètement l'interface et les données"""
        self.data = None
        self.raw_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {}
        self.results = {}
        self.best_model = None
        
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')
        
        for widget in self.figure_frame.winfo_children():
            widget.destroy()
        
        self.model_combobox.set('')
        self.model_combobox['values'] = []
        
        self.log_message("\n🔄 Interface réinitialisée - Prête pour une nouvelle base de données!")

    def confirm_reset(self):
        if messagebox.askyesno("Confirmation", 
                              "Voulez-vous vraiment tout réinitialiser?\nToutes les données et résultats seront perdus!"):
            self.reset_interface()

    def log_message(self, message):
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.configure(state='disabled')
        self.results_text.see(tk.END)

    def load_dataset(self):
        if self.data is not None:
            if not messagebox.askyesno("Confirmation", 
                                      "Un dataset est déjà chargé!\nVoulez-vous le remplacer?"):
                return
        
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.raw_data = pd.read_csv(file_path, sep=';', encoding='utf-8')
            self.data = self.raw_data.copy()
            self.log_message(f"\n✅ Dataset chargé : {file_path}")
            self.log_message(f"Shape du dataset : {self.data.shape}")

    def acp_analysis(self):
        if self.data is None:
            messagebox.showerror("Erreur", "Veuillez charger un dataset d'abord.")
            return
        
        try:
            # Préparation des données
            X = self.data.iloc[:, :-1].copy()
            y = self.data.iloc[:, -1] if len(self.data.columns) > 1 else None
            
            # Encodage des variables catégorielles
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Normalisation
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            
            # Tableau statistique
            desc = X.describe().T
            stats_table = desc[['mean', 'std', 'min', 'max']]
            stats_table.loc['Moyenne'] = stats_table.mean()
            
            self.log_message("\n📊 Tableau Statistique des Variables:")
            self.log_message(stats_table.to_string())
            
            # Matrice de covariance
            cov_matrix = pd.DataFrame(X_std).cov()
            self.log_message("\n📈 Matrice de Covariance:")
            self.log_message(str(cov_matrix))
            
            # Matrice de corrélation
            corr_matrix = pd.DataFrame(X_std).corr()
            self.log_message("\n📉 Matrice de Corrélation:")
            self.log_message(str(corr_matrix))
            
            # Analyse ACP
            pca = PCA()
            X_pca = pca.fit_transform(X_std)
            
            # Valeurs propres
            self.log_message("\n🔢 Valeurs Propres:")
            for i, val in enumerate(pca.explained_variance_):
                self.log_message(f"Composante {i+1}: {val:.4f}")
            
            # Vecteurs propres
            self.log_message("\n🧮 Vecteurs Propres:")
            for i, vec in enumerate(pca.components_):
                self.log_message(f"Composante {i+1}: {np.round(vec, 4)}")
            
            # Projection
            self.log_message("\n🎯 Projection sur les 2 premières composantes:")
            projection = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
            self.log_message(projection.head().to_string())
            
            # Visualisation
            fig = plt.figure(figsize=(12,10), facecolor=AppStyle.BACKGROUND)
            ax = fig.add_subplot(111)
            
            if y is not None and len(np.unique(y)) > 1:
                scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
            
            ax.set_xlabel('PC1', color=AppStyle.FOREGROUND)
            ax.set_ylabel('PC2', color=AppStyle.FOREGROUND)
            ax.set_title('Projection ACP', color=AppStyle.FOREGROUND)
            ax.grid(color=AppStyle.SECONDARY, linestyle='--', alpha=0.3)
            
            for spine in ax.spines.values():
                spine.set_color(AppStyle.FOREGROUND)
            ax.tick_params(colors=AppStyle.FOREGROUND)
            
            # Affichage dans l'interface
            for widget in self.figure_frame.winfo_children():
                widget.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ACP: {str(e)}")

    def preprocess(self):
        if self.data is not None:
            try:
                data_copy = self.data.copy()
                X = data_copy.iloc[:, :-1]
                y = data_copy.iloc[:, -1]

                for col in X.select_dtypes(include=['object', 'category']).columns:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

                imputer = SimpleImputer(strategy='mean')
                X_imputed = imputer.fit_transform(X)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)
                
                y_encoded = LabelEncoder().fit_transform(y)
                
                self.data = pd.DataFrame(X_scaled, columns=X.columns)
                self.data['Target'] = y_encoded
                
                self.log_message("\n✅ Prétraitement terminé avec succès!")
                self.log_message(f"- Encodage des caractéristiques catégorielles")
                self.log_message(f"- Imputation des valeurs manquantes (moyenne)")
                self.log_message(f"- Normalisation StandardScaler")
                self.log_message(f"- Encodage de la variable cible")
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du prétraitement: {str(e)}")

    def check_balance(self):
        if self.data is not None:
            class_counts = self.data['Target'].value_counts()
            threshold = 0.4
            balanced = (class_counts.max() - class_counts.min()) / class_counts.min() < threshold
            
            if not balanced:
                X = self.data.iloc[:, :-1]
                y = self.data['Target']
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                self.data = pd.DataFrame(X_res, columns=X.columns)
                self.data['Target'] = y_res
                self.log_message("\n✅ Données rééquilibrées avec SMOTE!")
            else:
                self.log_message("\nℹ️ Dataset déjà équilibré")

    def split_data(self):
        if self.data is not None:
            X = self.data.iloc[:, :-1]
            y = self.data['Target']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
            self.log_message("\n✅ Split Train/Test effectué (70/30)")
            self.log_message(f"Taille train set: {self.X_train.shape[0]} échantillons")
            self.log_message(f"Taille test set: {self.X_test.shape[0]} échantillons")


    def run_models(self):
        if self.X_train is None or self.y_train is None:
            messagebox.showerror("Erreur", "Veuillez préparer le dataset d'abord.")
            return

        self.models = {
            "Naive Bayes": (GaussianNB(), {}),
            "K-Nearest Neighbors": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
            "Decision Tree": (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
            "Random Forest": (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
            "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
            "K-Means": (KMeans(n_clusters=len(np.unique(self.y_train)), random_state=42), {})
        }

        self.results = {}
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for name, (model, params) in self.models.items():
            self.log_message(f"\n🔄 Entraînement de {name}")
            start_time = time.time()

            if name == "K-Means":
                model.fit(self.X_train)
                y_pred = model.predict(self.X_test)
                labels = np.zeros_like(y_pred)
                for i in range(len(np.unique(model.labels_))):
                    mask = (model.labels_ == i)
                    if np.any(mask):
                        majority_class = mode(self.y_train[mask], keepdims=False).mode
                        labels[model.predict(self.X_test) == i] = majority_class
                y_pred = labels
                best_model = model
            else:
                if params:
                    grid = GridSearchCV(model, params, cv=kf, scoring='accuracy', n_jobs=-1)
                    grid.fit(self.X_train, self.y_train)
                    best_model = grid.best_estimator_
                    self.log_message(f"🎯 Meilleurs paramètres : {grid.best_params_}")
                else:
                    best_model = model.fit(self.X_train, self.y_train)
                y_pred = best_model.predict(self.X_test)

            end_time = time.time()
            exec_time = end_time - start_time

            average_type = 'binary' if len(np.unique(self.y_test)) == 2 else 'macro'

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average=average_type, zero_division=0)
            recall = recall_score(self.y_test, y_pred, average=average_type, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average=average_type, zero_division=0)
            cm = confusion_matrix(self.y_test, y_pred)

            if name != "K-Means":
                cross_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=kf, scoring='accuracy')
                cv_score = np.mean(cross_scores)
            else:
                cv_score = None

            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'execution_time': exec_time,
                'model': best_model,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'average_type': average_type,
                'cv_score': cv_score
            }

            if cv_score is not None:
                self.log_message(f"✅ {name} - Accuracy: {accuracy:.2f} | CV: {cv_score:.2f}")
            else:
                self.log_message(f"✅ {name} - Accuracy: {accuracy:.2f}")
            self.log_message(f"   Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f} | Temps: {exec_time:.2f}s")

        self.model_combobox['values'] = list(self.results.keys())
        self.model_combobox.set("Choisis un modèle")
        messagebox.showinfo("Info", "Entraînement terminé!")

    def select_best_model(self):
        if self.results:
            self.best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            best_model_info = f"\n🏆 Meilleur modèle: {self.best_model[0]} (Accuracy: {self.best_model[1]['accuracy']:.2f})"
            self.log_message(best_model_info)

    def evaluate_model(self):
        if not self.results:
            messagebox.showerror("Erreur", "Entraînez d'abord les modèles")
            return

        model_name = self.model_var.get()
        if model_name not in self.results:
            messagebox.showerror("Erreur", "Modèle invalide!")
            return

        # Nettoyer la zone d'affichage graphique
        for widget in self.figure_frame.winfo_children():
            widget.destroy()

        result = self.results[model_name]
        model = result['model']
        y_pred = result['y_pred']
        cm = result['confusion_matrix']
        average_type = result['average_type']

        fig = plt.figure(figsize=(12,10), facecolor=AppStyle.BACKGROUND)
        fig.suptitle(f"Performance de {model_name}", color=AppStyle.FOREGROUND)

        ax1 = fig.add_subplot(121)
        if average_type == 'binary':
            try:
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(self.X_test)[:, 1]
                else:
                    y_score = model.decision_function(self.X_test)
                
                fpr, tpr, _ = roc_curve(self.y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax1.plot(fpr, tpr, color=AppStyle.ACCENT, label=f'AUC = {roc_auc:.2f}')
                ax1.plot([0, 1], [0, 1], color=AppStyle.SECONDARY, linestyle='--')
                ax1.set_xlabel('False Positive Rate', color=AppStyle.FOREGROUND)
                ax1.set_ylabel('True Positive Rate', color=AppStyle.FOREGROUND)
                ax1.legend()
            except Exception as e:
                ax1.text(0.5, 0.5, 'Non disponible', color=AppStyle.FOREGROUND)
        else:
            ax1.text(0.5, 0.5, 'ROC pour classification binaire', color=AppStyle.FOREGROUND)
        
        ax2 = fig.add_subplot(122)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Prédictions', color=AppStyle.FOREGROUND)
        ax2.set_ylabel('Vérités', color=AppStyle.FOREGROUND)

        for ax in [ax1, ax2]:
            ax.set_facecolor(AppStyle.TEXT_BG)
            ax.tick_params(colors=AppStyle.FOREGROUND)
            for spine in ax.spines.values():
                spine.set_color(AppStyle.FOREGROUND)

        canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        metrics = (f"\n📊 Evaluation de {model_name}:\n"
                f"Accuracy: {result['accuracy']:.2f}\n"
                f"Precision: {result['precision']:.2f}\n"
                f"Recall: {result['recall']:.2f}\n"
                f"F1-Score: {result['f1']:.2f}\n"
                f"Temps: {result['execution_time']:.2f}s")
        self.log_message(metrics)
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Projet de Classification Intelligente")
    root.geometry("1366x768")
    create_welcome_screen(root)
    root.mainloop()