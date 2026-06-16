# ==============================================================================
# PROYECTO: PREDICCIÓN DE CALIDAD DE VINO (RANDOM FOREST)
# ==============================================================================

# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configuración de estilo
sns.set(style="whitegrid")

# 2. CARGA DE DATOS
# Asegúrate de tener el archivo 'winequality-red.csv' en tu directorio
try:
    df = pd.read_csv('winequality-red.csv')
    print("Datos cargados correctamente.")
except FileNotFoundError:
    print("Error: El archivo 'winequality-red.csv' no se encuentra en el directorio.")

# 3. PREPROCESAMIENTO
# Creamos una variable binaria: 1 si es alta calidad (>=7), 0 de lo contrario
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Seleccionamos las variables predictoras (X) y la variable objetivo (y)
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# División en entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. ENTRENAMIENTO DEL MODELO
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. EVALUACIÓN
y_pred = rf_model.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred))

# 6. VISUALIZACIÓN
# Matriz de Confusión
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.show()

# Importancia de Características
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='magma')
plt.title('Importancia de Variables Fisicoquímicas')
plt.show()
