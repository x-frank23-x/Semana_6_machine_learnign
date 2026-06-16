import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. CARGA DE DATOS
file_path = r'C:\Users\FRANK\Documents\semana_4\winequality-red.csv'

# Intentamos cargar detectando el separador automáticamente
df = pd.read_csv(file_path, sep=None, engine='python')

# LIMPIEZA DE COLUMNAS: Elimina espacios extra en los nombres
df.columns = df.columns.str.strip()

print("Columnas detectadas:", df.columns.tolist())

# 2. PREPROCESAMIENTO
# Creamos la variable objetivo binaria
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Separamos X e y
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ENTRENAMIENTO
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. EVALUACIÓN
y_pred = rf_model.predict(X_test)
print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# 5. VISUALIZACIÓN
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=importances, y=importances.index)
plt.title('Importancia de variables')
plt.show()