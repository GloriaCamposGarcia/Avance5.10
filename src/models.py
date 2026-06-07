import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Grids de hiperparámetros
GRID_LOGISTIC = {'model__C': [0.01, 0.1, 1.0]}
GRID_TREE = {'model__max_depth': [3, 5, 7]}
GRID_RF = {'model__n_estimators': [50, 100]}
GRID_GB = {'model__n_estimators': [50, 100]}
GRID_SVM = {'model__C': [0.1, 1.0]}
GRID_KNN = {'model__n_neighbors': [3, 5, 7]}
GRID_DUMMY = {}

def build_preprocessor(numeric_features, categorical_features):
    """Construye un ColumnTransformer estandarizado para numérica y categórica."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'
    )
    return preprocessor

# Constructores de modelos baseline
def build_logistic():
    return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

def build_tree():
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)

def build_rf():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

def build_gb():
    return GradientBoostingClassifier(n_estimators=100, random_state=42)

def build_svm():
    return SVC(probability=True, random_state=42)

def build_knn():
    return KNeighborsClassifier(n_neighbors=5)

def build_dummy():
    return DummyClassifier(strategy='most_frequent')

# Auxiliares de evaluación
def evaluate_model(name, estimator, preprocessor, X_train, y_train, X_val, y_val, X_test, y_test):
    """Entrena y evalúa un estimador sobre splits de validación y prueba."""
    model = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
    model.fit(X_train, y_train)

    out = []
    for split_name, X_split, y_split in [('validation', X_val, y_val), ('test', X_test, y_test)]:
        y_pred = model.predict(X_split)
        y_score = model.predict_proba(X_split)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calcular ROC AUC de manera segura para evitar fallos con predicciones constantes
        roc_auc = np.nan
        if y_score is not None:
            try:
                roc_auc = round(float(roc_auc_score(y_split, y_score)), 4)
            except Exception:
                pass

        row = {
            'model': name,
            'split': split_name,
            'accuracy': round(float(accuracy_score(y_split, y_pred)), 4),
            'precision': round(float(precision_score(y_split, y_pred, zero_division=0)), 4),
            'recall_pos': round(float(recall_score(y_split, y_pred, zero_division=0)), 4),
            'f1': round(float(f1_score(y_split, y_pred, zero_division=0)), 4),
            'roc_auc': roc_auc,
        }
        out.append(row)
    return model, out

def quality_metrics(model, X_split, y_split):
    """Calcula las métricas operativas completas del modelo."""
    y_pred = model.predict(X_split)
    y_score = model.predict_proba(X_split)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Manejo defensivo en el cálculo de ROC AUC y PR AUC para evitar caídas con modelos dummy o constantes
    roc_auc = np.nan
    if y_score is not None:
        try:
            roc_auc = float(roc_auc_score(y_split, y_score))
        except Exception:
            pass
            
    pr_auc = np.nan
    if y_score is not None:
        try:
            pr_auc = float(average_precision_score(y_split, y_score))
        except Exception:
            pass

    out = {
        'precision': float(precision_score(y_split, y_pred, zero_division=0)),
        'recall_pos': float(recall_score(y_split, y_pred, zero_division=0)),
        'f1': float(f1_score(y_split, y_pred, zero_division=0)),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': confusion_matrix(y_split, y_pred).tolist(),
    }
    return out

def _base_feature_name(transformed_name: str) -> str:
    """Extrae el nombre base de la variable quitando el prefijo del ColumnTransformer."""
    parts = transformed_name.split('__')
    return parts[-1] if len(parts) > 1 else transformed_name

def max_priority_score(review_queue_json_text):
    """Analiza la cola de revisión en JSON y asigna un puntaje de prioridad basado en pesos."""
    weights = {'low': 0.4, 'medium': 1.0, 'high': 1.5}
    try:
        items = json.loads(review_queue_json_text)
        if not items:
            return 0.0
        return max(weights.get(it.get('priority', '').lower(), 0.0) for it in items)
    except Exception:
        return 0.0
