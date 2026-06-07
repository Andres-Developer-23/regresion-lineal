import io
import os
import base64
import uuid
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from django.conf import settings

sns.set_style('whitegrid')
sns.set_palette('muted')


def load_csv(path, delimiter=','):
    df = pd.read_csv(path, delimiter=delimiter)
    return df


def get_columns(df):
    return list(df.columns)


def get_numeric_columns(df):
    return list(df.select_dtypes(include=[np.number]).columns)


def get_unique_values(df, column, max_display=100):
    vals = df[column].unique()
    if len(vals) > max_display:
        return [str(v) for v in vals[:max_display]] + ['...']
    return [str(v) for v in vals]


def prepare_data(df, target_column, test_size=0.2, random_state=42, regression_type='logistic'):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X = X.astype(float)

    X = X.fillna(X.mean())

    if regression_type == 'linear':
        y = y.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), None, None, scaler

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2

    stratify_param = y_encoded if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), le.classes_.astype(str).tolist(), le, scaler


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=100, random_state=42):
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_linear_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {}
    metrics['r2'] = round(r2_score(y_test, y_pred), 4)
    metrics['mse'] = round(mean_squared_error(y_test, y_pred), 4)
    metrics['rmse'] = round(np.sqrt(metrics['mse']), 4)
    metrics['mae'] = round(mean_absolute_error(y_test, y_pred), 4)

    residuals = (y_test - y_pred).tolist()
    metrics['residuals'] = {
        'values': residuals[:500],
        'predicted': y_pred[:500].tolist(),
    }

    return metrics, y_pred, residuals


def generate_residuals_plot(y_test, y_pred):
    residuals = y_test - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_pred, residuals, alpha=0.6, color='#3498db', edgecolors='k', linewidth=0.5)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Valores predichos')
    ax1.set_ylabel('Residuales (real - predicción)')
    ax1.set_title('Residuales vs Predichos')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=30, color='#3498db', edgecolor='white', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Residuales')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_predicted_vs_actual_plot(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.6, color='#2ecc71', edgecolors='k', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Predicción perfecta')
    ax.set_xlabel('Valores reales')
    ax.set_ylabel('Valores predichos')
    ax.set_title('Valores Reales vs Predichos')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_model_params(model, feature_names, class_names):
    coef_data = {}
    if model.coef_.ndim == 2 and model.coef_.shape[0] > 1:
        for i, class_name in enumerate(class_names):
            coef_data[class_name] = {
                feature_names[j]: float(model.coef_[i][j])
                for j in range(len(feature_names))
            }
    else:
        class_name = str(class_names[1]) if len(class_names) > 1 else str(class_names[0])
        coef_data[class_name] = {
            feature_names[j]: float(model.coef_[0][j])
            for j in range(len(feature_names))
        }

    return {
        'coefficients': coef_data,
        'intercept': float(model.intercept_[0]) if model.intercept_.ndim == 1 else model.intercept_.tolist(),
        'classes': class_names,
        'feature_names': feature_names,
    }


def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    metrics = {}

    metrics['accuracy'] = round(accuracy_score(y_test, y_pred), 4)

    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            'precision': round(report[class_name]['precision'], 4),
            'recall': round(report[class_name]['recall'], 4),
            'f1_score': round(report[class_name]['f1-score'], 4),
            'support': int(report[class_name]['support']),
        }
    metrics['per_class'] = per_class

    if len(class_names) == 2 and y_prob is not None:
        try:
            auc_score = roc_auc_score(y_test, y_prob[:, 1])
            metrics['roc_auc'] = round(auc_score, 4)

            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            metrics['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
            }
        except Exception:
            pass

    return metrics, y_pred, y_prob


def generate_confusion_matrix_plot(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Matriz de Confusión')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


def generate_coefficients_plot(coef_data, feature_names, class_names):
    n_classes = len(class_names)

    def _plot_coefficients(ax, coefs, feature_names, title, show_legend=False):
        colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in coefs]
        y_pos = range(len(feature_names))

        for i, (c, color) in enumerate(zip(coefs, colors)):
            ax.plot([0, c], [i, i], color=color, linewidth=2.2, alpha=0.8, zorder=2)
            marker = 'o' if c != 0 else 's'
            ax.plot(c, i, marker=marker, color=color, markersize=10, zorder=3, markeredgecolor='white', markeredgewidth=1)

        max_abs = max(abs(c) for c in coefs) if coefs else 1
        offset_factor = max_abs * 0.05

        for i, c in enumerate(coefs):
            if c >= 0:
                ax.text(c + offset_factor, i, f'{c:.4f}',
                        va='center', ha='left', fontsize=10, fontweight='bold', color='#2ecc71')
            else:
                ax.text(c - offset_factor, i, f'{c:.4f}',
                        va='center', ha='right', fontsize=10, fontweight='bold', color='#e74c3c')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=11)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(0, ax.get_ylim()[0] - 0.3, '← Sin efecto', ha='center', va='top',
                fontsize=9, color='gray', style='italic')
        ax.set_xlim(-max_abs * 1.5, max_abs * 1.5)
        ax.set_xlabel('Coeficiente (log-odds)\n← Disminuye probabilidad     Aumenta probabilidad →',
                      fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(axis='x', alpha=0.3)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)

        if show_legend:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#2ecc71', linewidth=2, marker='o', markersize=8,
                       label='Coeficiente + (aumenta\nprobabilidad de la clase)'),
                Line2D([0], [0], color='#e74c3c', linewidth=2, marker='o', markersize=8,
                       label='Coeficiente - (disminuye\nprobabilidad de la clase)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                      framealpha=0.9, edgecolor='gray')

    if n_classes == 2:
        fig, ax = plt.subplots(figsize=(11, max(5, len(feature_names) * 0.55)))
        key = str(class_names[1])
        coefs = [coef_data[key][f] for f in feature_names]
        _plot_coefficients(ax, coefs, feature_names,
                           f'Coeficientes (log-odds) para clasificar como "{key}"', show_legend=True)
    else:
        fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, max(5, len(feature_names) * 0.55)))
        if n_classes == 1:
            axes = [axes]
        for i, class_name in enumerate(class_names):
            ax = axes[i]
            coefs = [coef_data[str(class_name)][f] for f in feature_names]
            _plot_coefficients(ax, coefs, feature_names,
                               f'Coeficientes para\nclase: {class_name}',
                               show_legend=(i == n_classes - 1))

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


def generate_roc_curve_plot(fpr, tpr, auc_score):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


def interpret_coefficient(coef, feature_name, scaled=True):
    odds_ratio = np.exp(abs(coef))
    if scaled:
        direction = 'aumenta' if coef > 0 else 'disminuye'
        arrow = '↑' if coef > 0 else '↓'
        if coef > 0:
            return (f'{arrow} Por cada desviación estándar de aumento en "{feature_name}", '
                    f'el log-odds de pertenecer a la clase aumenta en {coef:.4f} '
                    f'(odds ratio = e^{coef:.4f} = {odds_ratio:.2f}).')
        elif coef < 0:
            return (f'{arrow} Por cada desviación estándar de aumento en "{feature_name}", '
                    f'el log-odds de pertenecer a la clase disminuye en {abs(coef):.4f} '
                    f'(odds ratio = e^{coef:.4f} = {odds_ratio:.2f}).')
        else:
            return f'— "{feature_name}" no tiene efecto en la predicción (coeficiente = 0).'
    else:
        if coef > 0:
            return f'+ Por cada unidad adicional en "{feature_name}", la probabilidad aumenta (coef = {coef:.4f}).'
        elif coef < 0:
            return f'- Por cada unidad adicional en "{feature_name}", la probabilidad disminuye (coef = {coef:.4f}).'
        else:
            return f'— "{feature_name}" no tiene efecto en la predicción.'


def save_model_file(model):
    filename = f'model_{uuid.uuid4()}.joblib'
    path = os.path.join(settings.MEDIA_ROOT, 'models', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return f'models/{filename}'


def save_model_and_scaler(model, scaler):
    filename = f'model_{uuid.uuid4()}.joblib'
    path = os.path.join(settings.MEDIA_ROOT, 'models', filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, path)
    return f'models/{filename}'


def load_model_file(path):
    full_path = os.path.join(settings.MEDIA_ROOT, path)
    data = joblib.load(full_path)
    if isinstance(data, dict) and 'model' in data:
        return data['model']
    return data


def load_model_and_scaler(path):
    full_path = os.path.join(settings.MEDIA_ROOT, path)
    data = joblib.load(full_path)
    if isinstance(data, dict) and 'model' in data and 'scaler' in data:
        return data['model'], data['scaler']
    return data, None


def generate_sigmoid_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    z = np.linspace(-10, 10, 300)
    sigmoid = 1 / (1 + np.exp(-z))

    ax.plot(z, sigmoid, 'b-', lw=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Umbral de decisión (0.5)')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='z = 0 (log-odds = 0)')
    ax.fill_between(z, 0, sigmoid, where=(z >= 0), color='green', alpha=0.1, label='Clase positiva')
    ax.fill_between(z, 0, sigmoid, where=(z < 0), color='red', alpha=0.1, label='Clase negativa')
    ax.set_xlabel('z = β₀ + β₁x₁ + β₂x₂ + ... (combinación lineal)', fontsize=12)
    ax.set_ylabel('Probabilidad P(y=1|x)', fontsize=12)
    ax.set_title('Función Sigmoide — De regresión lineal a probabilidad de clasificación', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_train_test_split_plot():
    np.random.seed(42)
    n_total = 150
    n_train = 120
    n_test = 30

    fig, ax = plt.subplots(figsize=(10, 2))
    colors_train = ['#3498db'] * n_train
    colors_test = ['#e74c3c'] * n_test

    ax.barh(0, n_train, left=0, height=0.6, color='#3498db', edgecolor='white', label=f'Entrenamiento ({n_train} muestras, {n_train/n_total:.0%})')
    ax.barh(0, n_test, left=n_train, height=0.6, color='#e74c3c', edgecolor='white', label=f'Prueba ({n_test} muestras, {n_test/n_total:.0%})')
    ax.set_xlim(0, n_total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Muestras')
    ax.set_title('Separación Entrenamiento / Prueba', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.text(n_train / 2, 0, 'Entrenamiento\n(aprender patrones)', ha='center', va='center', color='white', fontweight='bold')
    ax.text(n_train + n_test / 2, 0, 'Prueba\n(evaluar desempeño)', ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_confusion_explanation_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_example = np.array([[50, 5], [8, 37]])

    sns.heatmap(cm_example, annot=False, fmt='d', cmap='Blues',
                xticklabels=['Pred: Negativo', 'Pred: Positivo'],
                yticklabels=['Real: Negativo', 'Real: Positivo'],
                ax=ax, cbar=False)

    ax.text(0.5, 0.5, 'VN\n50\nVerdaderos\nNegativos', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(1.5, 0.5, 'FP\n5\nFalsos\nPositivos\n(Error tipo I)', ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(0.5, 1.5, 'FN\n8\nFalsos\nNegativos\n(Error tipo II)', ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(1.5, 1.5, 'VP\n37\nVerdaderos\nPositivos', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    ax.set_title('Estructura de la Matriz de Confusión', fontsize=14)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_regularization_plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)
    n_points = 30

    scenarios = [
        {'C': 0.01, 'title': 'C = 0.01\n(mucha regularización)\nModelo simple', 'color': 'orange'},
        {'C': 1.0, 'title': 'C = 1.0\n(regularización estándar)\nBalance', 'color': 'green'},
        {'C': 100, 'title': 'C = 100\n(poca regularización)\nModelo complejo', 'color': 'red'},
    ]

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        X_2d = np.random.randn(n_points, 2)
        y_2d = (X_2d[:, 0] ** 2 + X_2d[:, 1] ** 2 > 1.5).astype(int)

        model = LogisticRegression(C=scenario['C'], max_iter=1000, random_state=42)
        model.fit(X_2d, y_2d)

        xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='coolwarm', edgecolors='k', s=50)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(scenario['title'], fontsize=11)
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')

    fig.suptitle('Efecto del parámetro C en la frontera de decisión', fontsize=14, y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_decision_boundary_plot(model, scaler, X_train_df, y_train, feature_names, class_names):
    if len(feature_names) != 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    X_plot = scaler.inverse_transform(X_train_df) if scaler else X_train_df.values
    X_plot_df = pd.DataFrame(X_plot, columns=feature_names)

    x_min, x_max = X_plot_df.iloc[:, 0].min() - 0.5, X_plot_df.iloc[:, 0].max() + 0.5
    y_min, y_max = X_plot_df.iloc[:, 1].min() - 0.5, X_plot_df.iloc[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=feature_names)

    if scaler:
        grid_scaled = scaler.transform(grid)
    else:
        grid_scaled = grid.values

    Z = model.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    contour = ax.contour(xx, yy, Z, colors='k', linewidths=2, levels=[0.5])

    for i, cls_name in enumerate(class_names):
        idx = y_train == i
        ax.scatter(X_plot_df.iloc[idx, 0], X_plot_df.iloc[idx, 1],
                   label=cls_name, s=60, edgecolors='k', linewidth=1, alpha=0.8)

    ax.set_xlabel(feature_names[0], fontsize=12)
    ax.set_ylabel(feature_names[1], fontsize=12)
    ax.set_title('Frontera de Decisión del Modelo', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
