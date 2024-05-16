import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos(ruta_datos):
    datos = []
    etiquetas = []
    categorias = os.listdir(ruta_datos)
    categoria_a_numero = {categoria: i for i, categoria in enumerate(categorias)}

    for categoria in categorias:
        ruta_categoria = os.path.join(ruta_datos, categoria)
        archivos_categoria = os.listdir(ruta_categoria)

        for archivo in archivos_categoria:
            ruta_archivo = os.path.join(ruta_categoria, archivo)
            data = np.load(ruta_archivo)
            datos.append(data)
            etiquetas.append(categoria_a_numero[categoria])

    longitud_minima = min(len(data) for data in datos)
    datos_recortados = [data[:longitud_minima] for data in datos]
    X = np.array(datos_recortados)
    y = np.array(etiquetas)
    return X, y


def entrenar_modelo(X_train, y_train):
    clf = SVC(kernel='linear')
    clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    return clf

def evaluar_modelo(clf, X_test, y_test):
    y_pred = clf.predict(X_test.reshape(X_test.shape[0], -1))
    precision = accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    reporte_clasificacion = classification_report(y_test, y_pred)
    return precision, matriz_confusion, reporte_clasificacion

def guardar_modelo(clf, ruta_modelo):
    joblib.dump(clf, ruta_modelo)


# Obtener la ruta actual
ruta_actual = os.getcwd()
ruta_raiz = os.path.dirname(os.path.dirname(os.path.dirname(ruta_actual)))
ruta_dataset = os.path.join(ruta_raiz, 'Data_Set')
ruta_datos = os.path.join(ruta_dataset, 'numpy_procesado')

X, y = cargar_datos(ruta_datos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = entrenar_modelo(X_train, y_train)
precision, matriz_confusion, reporte_clasificacion = evaluar_modelo(clf, X_test, y_test)

print("Precisión del clasificador:", precision)

print("\nReporte de Clasificación:")
print(reporte_clasificacion)

ruta_classifier= os.path.dirname(ruta_actual)
ruta_modelo = os.path.join(ruta_classifier, 'Models', 'modelo_entrenado.pkl')
guardar_modelo(clf, ruta_modelo)

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusion, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión')
plt.show()