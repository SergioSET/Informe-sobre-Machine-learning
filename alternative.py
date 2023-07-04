import graphviz
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.model_selection import cross_val_score

datos = pd.read_csv("stroke.csv")
LONGITUD = len(datos)

datosEntrenamiento = int(LONGITUD * 0.8)  # 80% para entrenar
datosPrueba = LONGITUD - datosEntrenamiento  # 20% para probar

train_data, test_data = sklearn.model_selection.train_test_split(
    datos, train_size=datosEntrenamiento, test_size=datosPrueba
)

categoricas = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

numericas = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

modificacionesPreProcesado = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categoricas),
        ("num", StandardScaler(), numericas),
    ]
)

X_train_processed = modificacionesPreProcesado.fit_transform(train_data)
y_train_processed = train_data["stroke"]


# Paso 4
hiperparametro = range(5, 51, 5)

arbolesDecisionPaso4 = []
for max_depth in hiperparametro:
    modeloPaso4 = tree.DecisionTreeClassifier(
        criterion="gini", max_depth=max_depth, splitter="best", random_state=123
    )
    modeloPaso4.fit(X_train_processed, y_train_processed)
    arbolesDecisionPaso4.append(modeloPaso4)

valoresPrecisionPaso4 = []
for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso4):
    precision = cross_val_score(
        tree, X_train_processed, y_train_processed, cv=5, scoring="accuracy"
    )
    valoresPrecisionPaso4.append(precision.mean())

tablaPaso4 = pd.DataFrame(
    {"Profundidad máxima": hiperparametro, "Precisión": valoresPrecisionPaso4}
)
print("Tabla de precisión de arboles punto 4")
print(tablaPaso4, "\n")

# # Paso 6
# arbolesDecisionPaso6 = []

# modeloPaso6 = DecisionTreeClassifier()
# for max_depth in hiperparametro:
#     modeloPaso6 = tree.DecisionTreeClassifier(
#         criterion="entropy", max_depth=max_depth, splitter="best", random_state=123
#     )
#     modeloPaso6.fit(X_train_processed, y_train_processed)
#     arbolesDecisionPaso6.append(modeloPaso6)

# valoresPrecisionPaso6 = []
# for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso6):
#     precision = cross_val_score(
#         tree, X_train_processed, y_train_processed, cv=5, scoring="accuracy"
#     )
#     valoresPrecisionPaso6.append(precision.mean())

# tablaPaso6 = pd.DataFrame(
#     {"Profundidad máxima": hiperparametro, "Precisión": valoresPrecisionPaso6}
# )
# print("Tabla de precisión de arboles punto 6")
# print(tablaPaso6, "\n")

# # Paso 8
# arbolesDecisionPaso8 = []

# for max_depth in hiperparametro:
#     modeloPaso8 = tree.DecisionTreeClassifier(
#         criterion="entropy", max_depth=max_depth, splitter="random", random_state=123
#     )
#     modeloPaso8.fit(X_train_processed, y_train_processed)
#     arbolesDecisionPaso8.append(modeloPaso8)

# valoresPrecisionPaso8 = []
# for max_depth, tree in zip(hiperparametro, arbolesDecisionPaso8):
#     precision = cross_val_score(
#         tree, X_train_processed, y_train_processed, cv=5, scoring="accuracy"
#     )
#     valoresPrecisionPaso8.append(precision.mean())

# tablaPaso8 = pd.DataFrame(
#     {"Profundidad máxima": hiperparametro, "Precisión": valoresPrecisionPaso8}
# )
# print("Tabla de precisión de arboles punto 8")
# print(tablaPaso8, "\n")

# modelo1 = tree.DecisionTreeClassifier(
#     criterion="gini", max_depth=5, splitter="best", random_state=123
# )
# modelo1.fit(X_train_processed, y_train_processed)
# scores1 = cross_val_score(
#     modelo1, X_train_processed, y_train_processed, cv=5, scoring="accuracy"
# )
# print(scores1)
# print(scores1.mean())

# tree.plot_tree(modelo1)
# tree.export_graphviz(decision_tree=modelo1, class_names=True,out_file="Arbol_Tuberculosis_1.dot")
