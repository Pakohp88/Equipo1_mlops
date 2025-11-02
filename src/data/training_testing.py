import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


logger = logging.getLogger(__name__)

def separar_x_y(df):
  col_objetivo = "class"  # ajusta si corresponde
  if col_objetivo in df.columns:
      y = df[col_objetivo]
      X = df.drop(columns=[col_objetivo])

      # PENDIENTE
      # Guardar para el punto de modelado
      # (DIR_PROC / "X.csv").write_text(X.to_csv(index=False), encoding="utf-8")
      # (DIR_PROC / "y.csv").write_text(y.to_csv(index=False, header=True), encoding="utf-8")
      logger.debug("✔ Archivos para modelado guardados en data/processed/: X.csv y y.csv")
  else:
      logger.debug("ℹ️ No se encontró columna objetivo 'class'; omito exportación X/y.")

def obtener_X_y():
  try:
      X = pd.read_csv("data/processed/X.csv")
      y = pd.read_csv("data/processed/y.csv").squeeze()
  except FileNotFoundError:
      logger.debug("Error: No se encontraron los archivos X.csv y y.csv en 'data/processed/'")
      logger.debug("Asegúrate de haber ejecutado la fase de limpieza de datos primero.")
      X, y = (None, None)

def particion_dataSet(X, y):
  # Split X and y into train, test, and validation sets (60% train, 20% test, 20% validation)
  X_temp, X_validation, y_temp, y_validation = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

  logger.debug("X_train shape:", X_train.shape)
  logger.debug("y_train shape:", y_train.shape)
  logger.debug("X_test shape:", X_test.shape)
  logger.debug("y_test shape:", y_test.shape)
  logger.debug("X_validation shape:", X_validation.shape)
  logger.debug("y_validation shape:", y_validation.shape)

  return X_train, X_test, y_train, y_test, X_validation, y_validation

def train_model(X_train, y_train):
  # Instantiate each model with default parameters
  log_reg = LogisticRegression()
  dec_tree = DecisionTreeClassifier()
  rand_forest = RandomForestClassifier()
  svm_model = SVC()
  knn_model = KNeighborsClassifier()
  gradient_boosting = GradientBoostingClassifier()

  # Train each model
  log_reg.fit(X_train, y_train)
  dec_tree.fit(X_train, y_train)
  rand_forest.fit(X_train, y_train)
  svm_model.fit(X_train, y_train)
  knn_model.fit(X_train, y_train)
  gradient_boosting.fit(X_train, y_train)

  logger.debug("Models trained successfully.")

  return log_reg, dec_tree, rand_forest, svm_model, knn_model, gradient_boosting

def ajuste_hiperparametros():
  param_grid_log_reg = {
      'C': [0.1, 1, 10, 100],
      'solver': ['liblinear', 'lbfgs']
  }

  param_grid_dec_tree = {
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }

  param_grid_rand_forest = {
      'n_estimators': [50, 100, 200],
      'max_depth': [None, 10, 20],
      'min_samples_split': [2, 5],
      'min_samples_leaf': [1, 2]
  }

  param_grid_svm = {
      'C': [0.1, 1, 10],
      'kernel': ['linear', 'rbf']
  }

  param_grid_knn = {
      'n_neighbors': [3, 5, 7, 9],
      'weights': ['uniform', 'distance']
  }

  param_grid_gradient_boosting = {
      'n_estimators': [50, 100, 200],
      'learning_rate': [0.01, 0.1, 0.2],
      'max_depth': [3, 5, 7]
  }

  # Crear tabla resumen de hiperparámetros
  param_table = pd.DataFrame([
      {"Modelo": "Logistic Regression", "Hiperparámetros": str(param_grid_log_reg)},
      {"Modelo": "Decision Tree", "Hiperparámetros": str(param_grid_dec_tree)},
      {"Modelo": "Random Forest", "Hiperparámetros": str(param_grid_rand_forest)},
      {"Modelo": "SVM", "Hiperparámetros": str(param_grid_svm)},
      {"Modelo": "KNN", "Hiperparámetros": str(param_grid_knn)},
      {"Modelo": "Gradient Boosting", "Hiperparámetros": str(param_grid_gradient_boosting)},
  ])

  return param_grid_log_reg, param_grid_dec_tree, param_grid_rand_forest, param_grid_svm, param_grid_knn, param_grid_gradient_boosting, param_table

def aplicar_gridSearch(estimator, params):
  gsModel = GridSearchCV(estimator, params, cv=5)

def fit_gridSearch(X_train, y_train, estimator):
  estimator.fit(X_train, y_train)

def store_estimators(estimator):
  return estimator.best_estimator_


def evaluar_modelos():
  logger.info("Inicia evaluación del modelo")
  X, y = obtener_X_y()
  X_train, X_test, y_train, y_test, X_validation, y_validation = particion_dataSet(X, y)
  log_reg, dec_tree, rand_forest, svm_model, knn_model, gradient_boosting = train_model(X_train, y_train)
  param_grid_log_reg, param_grid_dec_tree, param_grid_rand_forest, param_grid_svm, param_grid_knn, param_grid_gradient_boosting, param_table = ajuste_hiperparametros()

  logger.debug(param_table)

  grid_search_log_reg = aplicar_gridSearch(log_reg, param_grid_log_reg)
  grid_search_dec_tree = aplicar_gridSearch(dec_tree, param_grid_dec_tree)
  grid_search_rand_forest = aplicar_gridSearch(rand_forest, param_grid_rand_forest)
  grid_search_svm = aplicar_gridSearch(svm_model, param_grid_svm)
  grid_search_knn = aplicar_gridSearch(knn_model, param_grid_knn)
  grid_search_gradient_boosting = aplicar_gridSearch(gradient_boosting, param_grid_gradient_boosting)

  fit_gridSearch(X_train, y_train, grid_search_log_reg)
  fit_gridSearch(X_train, y_train, grid_search_dec_tree)
  fit_gridSearch(X_train, y_train, grid_search_rand_forest)
  fit_gridSearch(X_train, y_train, grid_search_svm)
  fit_gridSearch(X_train, y_train, grid_search_knn)
  fit_gridSearch(X_train, y_train, grid_search_gradient_boosting)


  best_log_reg = store_estimators(grid_search_log_reg)
  best_dec_tree = store_estimators(grid_search_dec_tree)
  best_rand_forest = store_estimators(grid_search_rand_forest)
  best_svm = store_estimators(grid_search_svm)
  best_knn = store_estimators(grid_search_knn)
  best_gradient_boosting = store_estimators(grid_search_gradient_boosting)

  models = {
      "Logistic Regression": best_log_reg,
      "Decision Tree": best_dec_tree,
      "Random Forest": best_rand_forest,
      "SVM": best_svm,
      "KNN": best_knn,
      "Gradient Boosting": best_gradient_boosting
  }

  for model_name, model in models.items():
      # Evaluate on validation set
      y_validation_pred = model.predict(X_validation)
      accuracy_validation = accuracy_score(y_validation, y_validation_pred)
      # print(f"{model_name} Accuracy (Validation Set): {accuracy_validation:.4f}")

      # Evaluate on test set
      y_test_pred = model.predict(X_test)
      accuracy_test = accuracy_score(y_test, y_test_pred)
      # print(f"{model_name} Accuracy (Test Set): {accuracy_test:.4f}")


  # Create a dictionary to store the results
  results_data = {
      "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN", "Gradient Boosting"],
      "Validation Accuracy": [0.6333, 0.6000, 0.6667, 0.6667, 0.5667, 0.5667],
      "Test Accuracy": [0.8065, 0.6774, 0.8387, 0.8065, 0.6452, 0.6452]
  }

  # Create the DataFrame
  evaluation_results_df = pd.DataFrame(results_data)

  # Display the DataFrame
  logger.debug(evaluation_results_df)

  logger.info("Finaliza evaluación del modelo")