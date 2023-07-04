import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Paso 1: Leer el archivo stroke.csv
datos = pd.read_csv('stroke.csv')

# Paso 2: Seleccionar aleatoriamente el 80% del conjunto de datos para entrenar y el 20% restante para las pruebas
X = datos.drop('stroke', axis=1)
y = datos['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% entrenamiento, 20% pruebas

# Paso 3: Utilizar una estrategia para normalizar los datos y llenar los datos faltantes
categoricas = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numericas = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Codificación one-hot para variables categóricas y escalado para variables numéricas
modificacionesPreProcesado = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categoricas),
        ('num', StandardScaler(), numericas)
    ])

X_train_processed = modificacionesPreProcesado.fit_transform(X_train)
X_test_processed = modificacionesPreProcesado.transform(X_test)

# Paso 4: Configurar los hiperparámetros del árbol de decisión de la siguiente manera: criterion=gini, splitter=best,
# y random_state=123. Obtener 10 árboles de decisión que resultan de modificar el hiperparámetro
# max_depth desde 5 hasta 50 con incrementos de 5

hiperparametro = range(5, 51, 5)

arbolesDecisionPaso4 = []
for max_depth in hiperparametro:
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    arbolesDecisionPaso4.append(classifier)

# Paso 5: Incluya en el notebook una tabla con el accuracy para los 10 árboles del punto anterior

valoresPrecisionPaso4 = []
for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso4):
    precision = tree.score(X_test_processed, y_test)
    valoresPrecisionPaso4.append(precision)

tablaPaso4 = pd.DataFrame({'Profundidad máxima': hiperparametro, 'Precisión': valoresPrecisionPaso4})
print('Tabla de precisión de arboles punto 4')
print(tablaPaso4, "\n")

# Paso 6: Repita el mismo procedimiento del punto 4 usando como hiperparámetros criterion=entropy, splitter=best,
# random_state=123, y variando el hiperparámetro max_depth desde 5 hasta 50 con incrementos de 5

arbolesDecisionPaso6 = []
for max_depth in hiperparametro:
    classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    arbolesDecisionPaso6.append(classifier)

# Paso 7: Incluya en el notebook una tabla con el accuracy para los 10 árboles del punto anterior

valoresPrecisionPaso6 = []
for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso6):
    accuracy = tree.score(X_test_processed, y_test)
    valoresPrecisionPaso6.append(accuracy)

tablaPaso6 = pd.DataFrame({'Profundidad máxima': hiperparametro, 'Precisión': valoresPrecisionPaso6})
print('Tabla de precisión de arboles punto 6')
print(tablaPaso6, "\n")

# Paso 8: Repita el mismo procedimiento del punto 4 usando como hiperparámetros criterion=entropy,
# splitter=random, random_state=123, y variando el hiperparámetro max_depth desde 5 hasta 50 con
# incrementos de 5

arbolesDecisionPaso8 = []
for max_depth in hiperparametro:
    classifier = HistGradientBoostingClassifier(max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    arbolesDecisionPaso8.append(classifier)

# Paso 9: Calcular y mostrar la tabla con los resultados de accuracy para los árboles de decisión con criterion=entropy y splitter=random

valoresPrecisionPaso8 = []
for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso8):
    accuracy = tree.score(X_test_processed, y_test)
    valoresPrecisionPaso8.append(accuracy)

tablaPaso8 = pd.DataFrame({'Profundidad máxima': hiperparametro, 'Precisión': valoresPrecisionPaso8})
print('Tabla de precisión de arboles punto 8')
print(tablaPaso8, "\n")