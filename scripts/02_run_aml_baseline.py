from sklearn.metrics import accuracy_score
import os
import sys
import json
import time
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
    quality_metrics, max_priority_score, build_stacking, build_voting
)

def main():
    # Intentar cargar variables de entorno desde un archivo .env local
    if HAS_DOTENV:
        try:
            load_dotenv(dotenv_path=project_root / '.env')
        except Exception:
            pass

    # 1. Definir rutas y cargar datasets
    source_results_path = project_root / 'data' / 'raw' / 'entity_source_results.csv'
    evidence_items_path = project_root / 'data' / 'raw' / 'evidence_items.csv'
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not source_results_path.exists() or not evidence_items_path.exists():
        print(f"ERROR: Falta alguno de los datasets en data/raw/")
        print("Por favor, coloca 'entity_source_results.csv' y 'evidence_items.csv' en la carpeta data/raw/")
        sys.exit(1)

    print(f"Cargando dataset de resultados desde {source_results_path}...")
    df_sources = pd.read_csv(source_results_path)
    print(f"Cargando dataset de evidencias desde {evidence_items_path}...")
    df_evidence = pd.read_csv(evidence_items_path)

    print("\nConsolidando métricas a nivel de entidad (OSINT Homogeneous)...")
    
    # Extraer catálogo de entidades únicas desde query_used
    entities = {}
    import ast
    for val in df_sources['query_used'].dropna().unique():
        try:
            data = ast.literal_eval(val)
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                ent_id = item.get('entity_id')
                name = item.get('query_value')
                metadata = item.get('metadata', {})
                country = metadata.get('country_code', '')
                raw_type = metadata.get('entity_type', '')
                ent_type = 'MORAL' if raw_type == 'PM' else ('FISICA' if raw_type == 'PF' else raw_type)
                
                if ent_id and name and ent_id not in entities:
                    entities[ent_id] = {
                        'entity_id': ent_id,
                        'entity_name': name,
                        'entity_type': ent_type,
                        'country_code': country
                    }
        except Exception:
            continue
            
    df_entities = pd.DataFrame(list(entities.values()))
    
    # Agrupaciones sobre entity_source_results.csv
    sources_eval = df_sources.groupby('entity_id').size().rename('sources_evaluated')
    sources_hallazgo = df_sources[df_sources['evidence_count'] > 0].groupby('entity_id').size().rename('sources_with_hallazgo')
    
    # Agrupaciones sobre evidence_items.csv
    max_id_score = df_evidence.groupby('entity_id')['identity_score'].max().rename('max_identity_score')
    evidence_cnt = df_evidence.groupby('entity_id').size().rename('evidence_items')
    
    df_evidence['is_review'] = df_evidence['review_required'].astype(str).str.lower().isin(['true', '1'])
    review_cnt = df_evidence[df_evidence['is_review']].groupby('entity_id').size().rename('review_items')
    
    def build_review_queue(group):
        reviews = []
        for _, r in group[group['is_review']].iterrows():
            p = 'high' if r['identity_score'] >= 0.8 else 'medium'
            reviews.append({'priority': p})
        return json.dumps(reviews)
        
    review_queues = df_evidence.groupby('entity_id').apply(build_review_queue).rename('review_queue_json')
    
    # Combinar todo
    df = df_entities.merge(sources_eval, on='entity_id', how='left')
    df = df.merge(sources_hallazgo, on='entity_id', how='left')
    df = df.merge(max_id_score, on='entity_id', how='left')
    df = df.merge(evidence_cnt, on='entity_id', how='left')
    df = df.merge(review_cnt, on='entity_id', how='left')
    df = df.merge(review_queues, on='entity_id', how='left')
    
    # Llenar nulos
    df['sources_with_hallazgo'] = df['sources_with_hallazgo'].fillna(0).astype(int)
    df['sources_evaluated'] = df['sources_evaluated'].fillna(0).astype(int)
    df['max_identity_score'] = df['max_identity_score'].fillna(0.0)
    df['evidence_items'] = df['evidence_items'].fillna(0).astype(int)
    df['review_items'] = df['review_items'].fillna(0).astype(int)
    df['review_queue_json'] = df['review_queue_json'].fillna('[]')
    
    # overall_decision
    df['overall_decision'] = np.where(
        df['review_items'] > 0,
        'needs_review',
        np.where(df['evidence_items'] > 0, 'accepted', 'no_match')
    )
    
    print(f"Dataset OSINT homogéneo consolidado con dimensiones: {df.shape}")

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
            t0 = time.time()
            best_pipe = trained_models[name]
            best_pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            
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
                'train_time_seconds': round(train_time, 4),
            })
            continue

        pipeline = Pipeline([('preprocessor', preprocessor), ('model', clone(candidate_models[name]))])
        
        t0 = time.time()
        search = GridSearchCV(pipeline, grid, scoring='average_precision', cv=cv, n_jobs=1, verbose=0)
        search.fit(X_train, y_train)
        train_time = time.time() - t0
        
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
            'train_time_seconds': round(train_time, 4),
        })

    # 7.1. Construcción de Estimadores de Ensamble Heterogéneo
    base_estimators = [
        ('logreg', clone(tuned_models['baseline_1_logreg'].named_steps['model'])),
        ('tree', clone(tuned_models['baseline_2_tree'].named_steps['model'])),
        ('rf', clone(tuned_models['baseline_3_rf'].named_steps['model'])),
    ]

    print("\nConstruyendo y entrenando Stacking Classifier (Ensamble Heterogéneo 1)...")
    try:
        t0_stack = time.time()
        stacking_clf = build_stacking(estimators=base_estimators)
        stacking_pipeline = Pipeline([('preprocessor', preprocessor), ('model', stacking_clf)])
        stacking_pipeline.fit(X_train, y_train)
        stacking_time = time.time() - t0_stack
        
        tuned_models['ensemble_stacking'] = stacking_pipeline
        metrics_val_stack = quality_metrics(stacking_pipeline, X_val, y_val)
        
        tuned_rows.append({
            'model': 'ensemble_stacking',
            'best_score_cv_pr_auc': np.nan,
            'accuracy': round(float(accuracy_score(y_val, stacking_pipeline.predict(X_val))), 4),
            'precision': round(float(metrics_val_stack['precision']), 4),
            'recall_pos': round(float(metrics_val_stack['recall_pos']), 4),
            'f1': round(float(metrics_val_stack['f1']), 4),
            'pr_auc': round(float(metrics_val_stack['pr_auc']), 4),
            'roc_auc': round(float(metrics_val_stack['roc_auc']), 4),
            'train_time_seconds': round(stacking_time, 4),
        })
        print("Stacking Classifier entrenado con éxito.")
    except Exception as e:
        print(f"Error al construir/entrenar el Stacking Classifier: {e}")

    print("\nConstruyendo y entrenando Voting Classifier (Ensamble Heterogéneo 2)...")
    try:
        t0_vote = time.time()
        voting_clf = build_voting(estimators=base_estimators, voting='soft')
        voting_pipeline = Pipeline([('preprocessor', preprocessor), ('model', voting_clf)])
        voting_pipeline.fit(X_train, y_train)
        voting_time = time.time() - t0_vote
        
        tuned_models['ensemble_voting'] = voting_pipeline
        metrics_val_vote = quality_metrics(voting_pipeline, X_val, y_val)
        
        tuned_rows.append({
            'model': 'ensemble_voting',
            'best_score_cv_pr_auc': np.nan,
            'accuracy': round(float(accuracy_score(y_val, voting_pipeline.predict(X_val))), 4),
            'precision': round(float(metrics_val_vote['precision']), 4),
            'recall_pos': round(float(metrics_val_vote['recall_pos']), 4),
            'f1': round(float(metrics_val_vote['f1']), 4),
            'pr_auc': round(float(metrics_val_vote['pr_auc']), 4),
            'roc_auc': round(float(metrics_val_vote['roc_auc']), 4),
            'train_time_seconds': round(voting_time, 4),
        })
        print("Voting Classifier entrenado con éxito.")
    except Exception as e:
        print(f"Error al construir/entrenar el Voting Classifier: {e}")

    # Exportar tabla comparativa ordenada por pr_auc desc
    tuned_df = pd.DataFrame(tuned_rows).sort_values('pr_auc', ascending=False).reset_index(drop=True)
    print("\nDesempeño de modelos ajustados en Validación:")
    print(tuned_df.to_string(index=False))
    
    metrics_comparison_path = processed_dir / 'metrics_comparison.csv'
    tuned_df.to_csv(metrics_comparison_path, index=False)
    print(f"Tabla comparativa guardada en: {metrics_comparison_path}")

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

    # 8.1. Generación de Gráficos de Diagnóstico (Mejores Prácticas AML)
    print("\nGenerando gráficos de diagnóstico...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
        from sklearn.inspection import permutation_importance

        # Estilo premium
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 14
        })

        # 1. Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Riesgo (0)', 'Riesgo (1)'],
                    yticklabels=['No Riesgo (0)', 'Riesgo (1)'], ax=ax)
        ax.set_title(f'Matriz de Confusión - {best_model_name} (Test)', pad=15, fontweight='bold')
        ax.set_ylabel('Clase Real', fontweight='bold')
        ax.set_xlabel('Clase Predicha', fontweight='bold')
        plt.tight_layout()
        cm_path = processed_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Matriz de confusión guardada en: {cm_path}")

        # 2. Curva ROC
        fig, ax = plt.subplots(figsize=(6, 5))
        if not np.isnan(y_score_test).all():
            fpr, tpr, _ = roc_curve(y_test, y_score_test)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc_val:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontweight='bold')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontweight='bold')
        ax.set_title(f'Curva ROC - {best_model_name} (Test)', pad=15, fontweight='bold')
        ax.legend(loc="lower right")
        plt.tight_layout()
        roc_path = processed_dir / 'roc_curve.png'
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Curva ROC guardada en: {roc_path}")

        # 3. Curva Precision-Recall
        fig, ax = plt.subplots(figsize=(6, 5))
        if not np.isnan(y_score_test).all():
            precision_pts, recall_pts, _ = precision_recall_curve(y_test, y_score_test)
            pr_auc_val = test_metrics.get('pr_auc', np.nan)
            if np.isnan(pr_auc_val):
                from sklearn.metrics import average_precision_score
                pr_auc_val = average_precision_score(y_test, y_score_test)
            ax.plot(recall_pts, precision_pts, color='forestgreen', lw=2, label=f'Curva PR (AUC = {pr_auc_val:.4f})')
        
        baseline_ratio = y_test.mean()
        ax.axhline(y=baseline_ratio, color='grey', linestyle='--', label=f'Línea Base ({baseline_ratio:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensibilidad)', fontweight='bold')
        ax.set_ylabel('Precision (Precisión)', fontweight='bold')
        ax.set_title(f'Curva Precision-Recall - {best_model_name} (Test)', pad=15, fontweight='bold')
        ax.legend(loc="lower left")
        plt.tight_layout()
        pr_path = processed_dir / 'precision_recall_curve.png'
        plt.savefig(pr_path, dpi=300)
        plt.close()
        print(f"Curva Precision-Recall guardada en: {pr_path}")

        # 4. Importancia de Características (Permutación)
        print("Calculando importancia de características mediante permutación...")
        fig, ax = plt.subplots(figsize=(7, 5))
        perm_result = permutation_importance(best_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
        sorted_idx = perm_result.importances_mean.argsort()
        
        ax.barh(range(len(sorted_idx)), perm_result.importances_mean[sorted_idx], 
                xerr=perm_result.importances_std[sorted_idx], align='center', color='royalblue', alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([X_test.columns[i] for i in sorted_idx], fontweight='bold')
        ax.set_xlabel('Disminución de Score (Exactitud / PR-AUC)', fontweight='bold')
        ax.set_title(f'Importancia de Características - {best_model_name}', pad=15, fontweight='bold')
        plt.tight_layout()
        feat_path = processed_dir / 'feature_importance.png'
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"Gráfico de importancia de características guardado en: {feat_path}")
        
    except Exception as e:
        print(f"Error al generar los gráficos de diagnóstico: {e}")
        import traceback
        traceback.print_exc()

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
