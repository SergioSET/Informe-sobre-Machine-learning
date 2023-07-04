import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Paso 1: Leer el archivo stroke.csv
data = pd.read_csv('stroke.csv')

# Paso 2: Dividir los datos en conjuntos de entrenamiento y pruebas
X = data.drop('stroke', axis=1)  # Reemplaza 'target_variable' con el nombre real de la variable objetivo
y = data['stroke']  # Reemplaza 'target_variable' con el nombre real de la variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 3: Preprocesamiento de variables categóricas y numéricas
categorical_features = ['gender']  # Reemplaza con las columnas categóricas presentes en tus datos
numeric_features = ['age', 'bmi', 'avg_glucose_level']  # Reemplaza con las columnas numéricas presentes en tus datos

# Codificación one-hot para variables categóricas y escalado para variables numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Paso 4: Configurar los hiperparámetros y obtener los árboles de decisión
hyperparameters = range(5, 51, 5)
decision_trees = []
for max_depth in hyperparameters:
    classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    decision_trees.append(classifier)

# Paso 5: Generar tabla de accuracy para los árboles de decisión

# Crear una lista para almacenar los resultados de accuracy
accuracy_scores = []

# Calcular y almacenar el accuracy de cada árbol de decisión
for max_depth, tree in zip(hyperparameters, decision_trees):
    accuracy = tree.score(X_test_processed, y_test)
    accuracy_scores.append(accuracy)

# Crear un DataFrame con los resultados
results_df = pd.DataFrame({'Max Depth': hyperparameters, 'Accuracy': accuracy_scores})

# Mostrar la tabla con los resultados
print(results_df)

# Paso 6: Configurar los hiperparámetros y obtener los árboles de decisión con criterion=entropy
decision_trees_entropy = []
for max_depth in hyperparameters:
    classifier = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    decision_trees_entropy.append(classifier)

# Paso 7: Calcular y mostrar la tabla con los resultados de accuracy para los árboles de decisión con criterion=entropy
accuracy_scores_entropy = []
for max_depth, tree in zip(hyperparameters, decision_trees_entropy):
    accuracy = tree.score(X_test_processed, y_test)
    accuracy_scores_entropy.append(accuracy)

results_df_entropy = pd.DataFrame({'Max Depth': hyperparameters, 'Accuracy': accuracy_scores_entropy})
print(results_df_entropy)

# Paso 8: Configurar los hiperparámetros y obtener los árboles de decisión con criterion=entropy y splitter=random
decision_trees_random = []
for max_depth in hyperparameters:
    classifier = HistGradientBoostingClassifier(max_depth=max_depth, random_state=123)
    classifier.fit(X_train_processed, y_train)
    decision_trees_random.append(classifier)

# Paso 9: Calcular y mostrar la tabla con los resultados de accuracy para los árboles de decisión con criterion=entropy y splitter=random
accuracy_scores_random = []
for max_depth, tree in zip(hyperparameters, decision_trees_random):
    accuracy = tree.score(X_test_processed, y_test)
    accuracy_scores_random.append(accuracy)

results_df_random = pd.DataFrame({'Max Depth': hyperparameters, 'Accuracy': accuracy_scores_random})
print(results_df_random)
