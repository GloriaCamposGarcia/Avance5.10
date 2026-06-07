from sklearn.metrics import accuracy_score
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Añadir la raíz del proyecto al path para poder importar de src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.models import (
    build_preprocessor, GRID_DUMMY, GRID_GB, GRID_KNN, GRID_LOGISTIC,
    GRID_RF, GRID_SVM, GRID_TREE, build_dummy, build_gb, build_knn,
    build_logistic, build_rf, build_svm, build_tree, evaluate_model,
    quality_metrics, max_priority_score
)

def main():
    # Intentar cargar variables de entorno desde un archivo .env local
    if HAS_DOTENV:
        try:
            load_dotenv(dotenv_path=project_root / '.env')
        except Exception:
            pass

    # 1. Definir rutas y cargar dataset
    raw_data_path = project_root / 'data' / 'raw' / 'entities_osint_homogeneous.csv'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_path.exists():
        print(f"ERROR: No se encontró el dataset en {raw_data_path}")
        print("Por favor, coloca 'entities_osint_homogeneous.csv' en la carpeta data/raw/")
        sys.exit(1)

    print(f"Cargando dataset OSINT desde {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Dataset cargado con dimensiones: {df.shape}")

    # 2. Generar variable objetivo
    target_col = 'riesgo_fraude_aml'
    q75_identity = df['max_identity_score'].quantile(0.75)
    
    df[target_col] = (
        (df['overall_decision'].eq('needs_review')) & (
            (df['sources_with_hallazgo'] >= 3)
            | (df['max_identity_score'] >= q75_identity)
            | (df['review_items'] >= 3)
        )
    ).astype(int)
    
    print(f"\nTarget '{target_col}' generado con éxito.")
    print("Distribución de clases:")
    print(df[target_col].value_counts(normalize=True).round(4).to_string())

    # 3. Filtrado de leakage y selección de características
    exclude_leakage = [
        'overall_decision', 'run_id', 'entity_id', 'entity_name', 
        'finding_summary', 'context_summary', 'source_breakdown_json', 
        'supporting_evidence_json', 'review_queue_json', 'error_log_json'
    ]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    excluded_features = sorted(set(exclude_leakage + constant_cols + [target_col]))
    candidate_features = [c for c in df.columns if c not in excluded_features]

    X = df[candidate_features].copy()
    y = df[target_col].copy()
    print(f"\nCaracterísticas candidatas seleccionadas ({len(candidate_features)}): {candidate_features}")

    # 4. Partición estratificada (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    # 5. Preprocesamiento Pipeline
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # 6. Entrenamiento y evaluación de baselines
    print("\nEntrenando estimadores baseline...")
    candidate_models = {
        'baseline_0_dummy': build_dummy(),
        'baseline_1_logreg': build_logistic(),
        'baseline_2_tree': build_tree(),
        'baseline_3_rf': build_rf(),
        'baseline_4_gb': build_gb(),
        'baseline_5_svm': build_svm(),
        'baseline_6_knn': build_knn(),
    }

    trained_models = {}
    model_rows = []
    for name, estimator in candidate_models.items():
        pipe, metrics = evaluate_model(name, estimator, preprocessor, X_train, y_train, X_val, y_val, X_test, y_test)
        # evaluate_model retorna una lista de métricas por split, tomamos la de validación para comparar
        val_metrics = [m for m in metrics if m['split'] == 'validation'][0]
        # quitamos split key
        val_metrics.pop('split')
        trained_models[name] = pipe
        model_rows.append(val_metrics)

    metrics_models_df = pd.DataFrame(model_rows).sort_values(['f1', 'roc_auc'], ascending=False).reset_index(drop=True)
    print("\nDesempeño de modelos baseline en Validación:")
    print(metrics_models_df.to_string(index=False))

    # 7. Ajuste de hiperparámetros (GridSearchCV)
    print("\nEjecutando GridSearchCV sobre baselines...")
    param_grids = {
        'baseline_0_dummy': GRID_DUMMY,
        'baseline_1_logreg': GRID_LOGISTIC,
        'baseline_2_tree': GRID_TREE,
        'baseline_3_rf': GRID_RF,
        'baseline_4_gb': GRID_GB,
        'baseline_5_svm': GRID_SVM,
        'baseline_6_knn': GRID_KNN,
    }

    tuned_models = {}
    tuned_rows = []
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name in metrics_models_df['model'].tolist():
        grid = param_grids.get(name, {})
        if not grid:
            # dummy no tiene grid
            best_pipe = trained_models[name]
            tuned_models[name] = best_pipe
            metrics_val = quality_metrics(best_pipe, X_val, y_val)
            tuned_rows.append({
                'model': name,
                'best_score_cv_pr_auc': np.nan,
                'accuracy': round(float(accuracy_score(y_val, best_pipe.predict(X_val))), 4),
                'precision': round(float(metrics_val['precision']), 4),
                'recall_pos': round(float(metrics_val['recall_pos']), 4),
                'f1': round(float(metrics_val['f1']), 4),
                'pr_auc': round(float(metrics_val['pr_auc']), 4),
                'roc_auc': round(float(metrics_val['roc_auc']), 4),
            })
            continue

        pipeline = Pipeline([('preprocessor', preprocessor), ('model', clone(candidate_models[name]))])
        search = GridSearchCV(pipeline, grid, scoring='average_precision', cv=cv, n_jobs=1, verbose=0)
        search.fit(X_train, y_train)
        
        best_pipe = search.best_estimator_
        tuned_models[name] = best_pipe
        
        # Evaluar
        metrics_val = quality_metrics(best_pipe, X_val, y_val)
        tuned_rows.append({
            'model': name,
            'best_score_cv_pr_auc': round(float(search.best_score_), 4),
            'accuracy': round(float(accuracy_score(y_val, best_pipe.predict(X_val))), 4),
            'precision': round(float(metrics_val['precision']), 4),
            'recall_pos': round(float(metrics_val['recall_pos']), 4),
            'f1': round(float(metrics_val['f1']), 4),
            'pr_auc': round(float(metrics_val['pr_auc']), 4),
            'roc_auc': round(float(metrics_val['roc_auc']), 4),
        })

    tuned_df = pd.DataFrame(tuned_rows).sort_values('pr_auc', ascending=False).reset_index(drop=True)
    print("\nDesempeño de modelos ajustados en Validación:")
    print(tuned_df.to_string(index=False))

    # 8. Selección final de modelo y exportación
    best_model_name = tuned_df.iloc[0]['model'] if not tuned_df.empty else metrics_models_df.iloc[0]['model']
    best_pipeline = tuned_models.get(best_model_name) or trained_models.get(best_model_name)
    
    print(f"\nModelo final seleccionado: {best_model_name}")

    # Serialización
    model_export_path = processed_dir / 'best_model.pkl'
    joblib.dump(best_pipeline, model_export_path)
    print(f"Modelo exportado en: {model_export_path}")

    # Evaluación final en prueba (test)
    y_pred_test = best_pipeline.predict(X_test)
    try:
        y_score_test = best_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_score_test = np.full(shape=len(y_pred_test), fill_value=np.nan, dtype=float)

    test_metrics = quality_metrics(best_pipeline, X_test, y_test)
    test_metrics_df = pd.DataFrame([{
        'model': best_model_name,
        'accuracy': round(float(accuracy_score(y_test, y_pred_test)), 4),
        'precision': round(test_metrics['precision'], 4),
        'recall': round(test_metrics['recall_pos'], 4),
        'f1': round(test_metrics['f1'], 4),
        'roc_auc': round(test_metrics['roc_auc'], 4),
        'pr_auc': round(test_metrics['pr_auc'], 4),
    }])
    
    test_metrics_path = processed_dir / 'test_metrics.csv'
    test_metrics_df.to_csv(test_metrics_path, index=False)
    print(f"Métricas sobre split de prueba guardadas en: {test_metrics_path}")

    # Guardar predicciones de test (y_true, y_pred, y_score)
    test_predictions_df = pd.DataFrame({
        'y_true': np.asarray(y_test).ravel(),
        'y_pred': np.asarray(y_pred_test).ravel(),
        'y_score': np.asarray(y_score_test).ravel(),
    })
    test_predictions_path = processed_dir / 'test_predictions.csv'
    test_predictions_df.to_csv(test_predictions_path, index=False)
    print(f"Predicciones de prueba guardadas en: {test_predictions_path}")

    # Manifest json
    manifest_path = processed_dir / 'run_manifest.json'
    manifest = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'project_root': str(project_root),
        'candidate_features': candidate_features,
        'selected_model': best_model_name,
        'model_path': str(model_export_path),
        'test_metrics_path': str(test_metrics_path),
        'test_predictions_path': str(test_predictions_path),
        'confusion_matrix': test_metrics['confusion_matrix']
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Manifest JSON exportado en: {manifest_path}")

    # 9. Inferencia y OSINT Risk Score continuo
    print("\nCalculando OSINT Risk Score operativo continuo...")
    df_scored = df.copy()
    df_scored['priority_score_raw'] = df_scored['review_queue_json'].apply(max_priority_score)
    
    eps = 1e-9
    df_scored['max_identity_norm'] = (df_scored['max_identity_score'] - df_scored['max_identity_score'].min()) / (df_scored['max_identity_score'].max() - df_scored['max_identity_score'].min() + eps)
    df_scored['hallazgo_ratio'] = df_scored['sources_with_hallazgo'] / (df_scored['sources_evaluated'] + eps)
    df_scored['evidence_norm'] = (df_scored['evidence_items'] - df_scored['evidence_items'].min()) / (df_scored['evidence_items'].max() - df_scored['evidence_items'].min() + eps)
    df_scored['priority_norm'] = (df_scored['priority_score_raw'] - df_scored['priority_score_raw'].min()) / (df_scored['priority_score_raw'].max() - df_scored['priority_score_raw'].min() + eps)
    
    df_scored['osint_risk_score'] = (0.50*df_scored['max_identity_norm'] + 0.20*df_scored['hallazgo_ratio'] + 0.20*df_scored['evidence_norm'] + 0.10*df_scored['priority_norm'])
    
    q1 = df_scored['osint_risk_score'].quantile(0.33)
    q2 = df_scored['osint_risk_score'].quantile(0.66)
    df_scored['risk_level'] = pd.cut(df_scored['osint_risk_score'], bins=[-np.inf, q1, q2, np.inf], labels=['bajo', 'medio', 'alto'])
    
    print("Distribución de niveles de riesgo operativo:")
    print(df_scored['risk_level'].value_counts().to_string())

    # Exportar predicciones
    risk_scores_path = processed_dir / 'osint_risk_scores.csv'
    df_scored[['entity_id', 'entity_name', 'osint_risk_score', 'risk_level']].to_csv(risk_scores_path, index=False)
    print(f"Predicciones y scores continuos exportados en: {risk_scores_path}")

    print("\nFase 2 completada con éxito.")

if __name__ == '__main__':
    main()
