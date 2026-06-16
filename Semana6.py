# ==============================================================================
# PROYECTO: CLASIFICADOR DE CALIDAD DE VINO - RANDOM FOREST
# Documentación técnica del pipeline de Machine Learning
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- FASE 1: CARGA DE DATOS Y NORMALIZACIÓN ---
# Se utiliza 'sep=None' para detectar automáticamente el formato (CSV, TSV, etc.)
# y 'engine=python' para asegurar compatibilidad robusta.
file_path = r'C:\Users\FRANK\Documents\semana_4\winequality-red.csv'
df = pd.read_csv(file_path, sep=None, engine='python')

# Limpieza: Eliminación de espacios en blanco residuales en los nombres de columnas
# Esto previene errores de acceso (KeyError) por formato de archivo inconsistente.
df.columns = df.columns.str.strip()

# --- FASE 2: PREPROCESAMIENTO ---
# Conversión de un problema multiclase a uno binario.
# Se etiqueta como '1' (Excelente) si calidad >= 7, else '0' (Estándar).
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Definición de variables predictoras (X) y variable objetivo (y).
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# División: 80% entrenamiento (aprendizaje) y 20% prueba (validación ciega).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- FASE 3: MODELADO Y ENTRENAMIENTO ---
# Instanciación del modelo: Random Forest.
# 'n_estimators=100' crea 100 árboles de decisión para mejorar la robustez.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- FASE 4: EVALUACIÓN ---
# Generación de predicciones sobre datos nunca antes vistos por el modelo.
y_pred = rf_model.predict(X_test)

# --- FASE 5: VISUALIZACIÓN E INTERPRETACIÓN ---
# Matriz de Confusión para comparar predicciones contra valores reales.
# Importancia de variables para identificar qué características definen la calidad.
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()
