import os
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.conf import settings
from .models import Dataset, TrainedModel
from .forms import DatasetUploadForm, TrainForm, PredictForm
from . import utils


def index(request):
    datasets_count = Dataset.objects.count()
    models_count = TrainedModel.objects.count()

    sigmoid_plot = utils.generate_sigmoid_plot()
    train_test_plot = utils.generate_train_test_split_plot()

    return render(request, 'classifier/index.html', {
        'datasets_count': datasets_count,
        'models_count': models_count,
        'sigmoid_plot': sigmoid_plot,
        'train_test_plot': train_test_plot,
    })


def datasets_list(request):
    datasets = Dataset.objects.all()
    return render(request, 'classifier/datasets.html', {
        'datasets': datasets,
    })


def upload_dataset(request):
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['file']
            delimiter = form.cleaned_data['delimiter']
            name = form.cleaned_data['name']
            description = form.cleaned_data.get('description', '')

            dataset = Dataset(name=name, file=file, description=description)
            dataset.save()

            try:
                file_path = dataset.file.path
                df = utils.load_csv(file_path, delimiter)
                dataset.rows = len(df)
                dataset.columns = len(df.columns)
                dataset.save()
                messages.success(request, f'Dataset "{name}" cargado correctamente ({dataset.rows} filas, {dataset.columns} columnas).')
                return redirect('dataset_detail', pk=dataset.pk)
            except Exception as e:
                dataset.delete()
                messages.error(request, f'Error al leer el archivo CSV: {str(e)}')
                return render(request, 'classifier/upload.html', {'form': form})
    else:
        form = DatasetUploadForm()
    return render(request, 'classifier/upload.html', {'form': form})


def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)

    try:
        file_path = dataset.file.path
        delimiter = ',' if dataset.filename().endswith('.csv') else ','
        df = utils.load_csv(file_path, delimiter)
    except Exception as e:
        messages.error(request, f'Error al leer el dataset: {str(e)}')
        return redirect('datasets')

    columns = utils.get_columns(df)
    numeric_cols = utils.get_numeric_columns(df)
    preview = df.head(10).to_dict('records')
    col_names = list(df.columns) if len(df) > 0 else []
    stats = df.describe().to_dict() if len(numeric_cols) > 0 else {}
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

    unique_vals = {}
    for col in columns[:5]:
        unique_vals[col] = utils.get_unique_values(df, col, max_display=10)

    models = TrainedModel.objects.filter(dataset=dataset)

    return render(request, 'classifier/dataset_detail.html', {
        'dataset': dataset,
        'columns': columns,
        'preview': preview,
        'col_names': col_names,
        'stats': stats,
        'dtypes': dtypes,
        'unique_vals': unique_vals,
        'models': models,
        'numeric_cols': numeric_cols,
    })


def train_model(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)

    try:
        file_path = dataset.file.path
        delimiter = ',' if dataset.filename().endswith('.csv') else ','
        df = utils.load_csv(file_path, delimiter)
    except Exception as e:
        messages.error(request, f'Error al leer el dataset: {str(e)}')
        return redirect('dataset_detail', pk=pk)

    if request.method == 'POST':
        form = TrainForm(request.POST, columns=utils.get_columns(df))
        if form.is_valid():
            target = form.cleaned_data['target_column']
            test_size = form.cleaned_data['test_size']
            regression_type = form.cleaned_data['regression_type']
            random_state = form.cleaned_data['random_state']
            c_param = form.cleaned_data.get('c_param', 1.0)
            max_iter = form.cleaned_data.get('max_iter', 200)

            try:
                n_unique = df[target].nunique()

                if regression_type == 'logistic':
                    if n_unique > 20:
                        raise ValueError(
                            f'La columna "{target}" tiene {n_unique} valores únicos. '
                            'Para clasificación, la variable objetivo debe tener pocas clases '
                            '(idealmente 2-10). Selecciona otra columna o verifica tus datos.'
                        )
                    if n_unique < 2:
                        raise ValueError(
                            f'La columna "{target}" tiene solo {n_unique} valor(es) único(s). '
                            'La variable objetivo debe tener al menos 2 clases distintas.'
                        )
                else:
                    if not np.issubdtype(df[target].dtype, np.number):
                        raise ValueError(
                            f'La columna "{target}" es de tipo {df[target].dtype}. '
                            'Para regresión lineal, la variable objetivo debe ser numérica continua.'
                        )
                    if n_unique < 5:
                        raise ValueError(
                            f'La columna "{target}" tiene solo {n_unique} valores únicos. '
                            'Para regresión lineal, la variable objetivo debe tener al menos 5 valores distintos.'
                        )

                X_train, X_test, y_train, y_test, feature_names, class_names, le, scaler = utils.prepare_data(
                    df, target, test_size, random_state, regression_type
                )

                if regression_type == 'linear':
                    model = utils.train_linear_regression(X_train, y_train)
                    metrics, y_pred, residuals = utils.evaluate_linear_model(model, X_test, y_test)
                    model_file_path = utils.save_model_and_scaler(model, scaler)

                    trained = TrainedModel(
                        dataset=dataset,
                        name=f'{dataset.name} - RL',
                        model_file=model_file_path,
                        features=feature_names,
                        target=target,
                        coefficients=None,
                        intercept=float(model.intercept_),
                        metrics=metrics,
                        params={
                            'test_size': test_size,
                            'random_state': random_state,
                        },
                        is_binary=False,
                        model_type='linear',
                        test_size=test_size,
                    )
                    trained.save()
                    messages.success(request, 'Modelo de regresión lineal entrenado correctamente.')
                    return redirect('model_detail', pk=trained.pk)
                else:
                    model = utils.train_logistic_regression(X_train, y_train, c_param, max_iter, random_state)
                    model_params = utils.get_model_params(model, feature_names, class_names)
                    model_file_path = utils.save_model_and_scaler(model, scaler)

                    trained = TrainedModel(
                        dataset=dataset,
                        name=f'{dataset.name} - LR (C={c_param})',
                        model_file=model_file_path,
                        features=feature_names,
                        target=target,
                        coefficients=model_params['coefficients'],
                        intercept=model_params['intercept'],
                        params={
                            'C': c_param,
                            'max_iter': max_iter,
                            'random_state': random_state,
                            'test_size': test_size,
                        },
                        classes=class_names,
                        is_binary=(len(class_names) == 2),
                        test_size=test_size,
                    )
                    trained.save()
                    messages.success(request, 'Modelo de regresión logística entrenado correctamente.')
                    return redirect('model_detail', pk=trained.pk)

            except Exception as e:
                messages.error(request, f'Error al entrenar el modelo: {str(e)}')
                regularization_plot = utils.generate_regularization_plot()
                return render(request, 'classifier/train.html', {
                    'form': form,
                    'dataset': dataset,
                    'regularization_plot': regularization_plot,
                })
    else:
        form = TrainForm(columns=utils.get_columns(df))

    regularization_plot = utils.generate_regularization_plot()

    return render(request, 'classifier/train.html', {
        'form': form,
        'dataset': dataset,
        'regularization_plot': regularization_plot,
    })


def models_list(request):
    models_qs = TrainedModel.objects.all()
    return render(request, 'classifier/models.html', {
        'models': models_qs,
    })


def model_detail(request, pk):
    trained = get_object_or_404(TrainedModel, pk=pk)
    dataset = trained.dataset

    if trained.model_type == 'linear':
        return render(request, 'classifier/model_detail.html', {
            'model': trained,
            'dataset': dataset,
        })

    interpretations = []
    for class_name, coef_dict in trained.coefficients.items():
        for feat, coef_val in coef_dict.items():
            interpretations.append({
                'class_name': class_name,
                'feature': feat,
                'coefficient': coef_val,
                'abs_coef': abs(coef_val),
                'interpretation': utils.interpret_coefficient(coef_val, feat),
            })

    interpretations.sort(key=lambda x: x['abs_coef'], reverse=True)

    max_abs = max((x['abs_coef'] for x in interpretations), default=1)

    top_features = {}
    for class_name in trained.classes:
        class_items = [x for x in interpretations if x['class_name'] == class_name]
        if class_items:
            top_features[class_name] = class_items[:3]

    classification_plot = utils.generate_confusion_explanation_plot()
    sigmoid_plot = utils.generate_sigmoid_plot()

    return render(request, 'classifier/model_detail.html', {
        'model': trained,
        'dataset': dataset,
        'interpretations': interpretations[:30],
        'max_abs_coef': max_abs,
        'top_features': top_features,
        'classification_plot': classification_plot,
        'sigmoid_plot': sigmoid_plot,
    })


def evaluate_model(request, pk):
    trained = get_object_or_404(TrainedModel, pk=pk)

    try:
        df = utils.load_csv(trained.dataset.file.path)

        random_state = trained.params.get('random_state', 42)
        X_train, X_test, y_train, y_test, feature_names, class_names, le, scaler = utils.prepare_data(
            df, trained.target, trained.test_size, random_state, trained.model_type
        )

        model = utils.load_model_file(trained.model_file.name)

        if trained.model_type == 'linear':
            metrics, y_pred, residuals = utils.evaluate_linear_model(model, X_test, y_test)
            residuals_plot = utils.generate_residuals_plot(y_test, y_pred)
            predicted_vs_actual_plot = utils.generate_predicted_vs_actual_plot(y_test, y_pred)

            return render(request, 'classifier/evaluate.html', {
                'model': trained,
                'metrics': metrics,
                'residuals_plot': residuals_plot,
                'predicted_vs_actual_plot': predicted_vs_actual_plot,
                'class_names': None,
                'feature_names': feature_names,
                'dataset': trained.dataset,
            })

        metrics, y_pred, y_prob = utils.evaluate_model(model, X_test, y_test, class_names)

        cm_plot = utils.generate_confusion_matrix_plot(
            metrics['confusion_matrix'], class_names
        )

        cm_explanation = utils.generate_confusion_explanation_plot()

        roc_plot = None
        if trained.is_binary and 'roc_curve' in metrics:
            roc_plot = utils.generate_roc_curve_plot(
                metrics['roc_curve']['fpr'],
                metrics['roc_curve']['tpr'],
                metrics['roc_auc']
            )

        return render(request, 'classifier/evaluate.html', {
            'model': trained,
            'metrics': metrics,
            'cm_plot': cm_plot,
            'cm_explanation': cm_explanation,
            'roc_plot': roc_plot,
            'class_names': class_names,
            'feature_names': feature_names,
            'dataset': trained.dataset,
        })

    except Exception as e:
        messages.error(request, f'Error al evaluar el modelo: {str(e)}')
        return redirect('model_detail', pk=pk)


def predict_view(request, pk):
    trained = get_object_or_404(TrainedModel, pk=pk)
    features = {f: float for f in trained.features}

    if request.method == 'POST':
        form = PredictForm(request.POST, features=features)
        if form.is_valid():
            try:
                model, scaler = utils.load_model_and_scaler(trained.model_file.name)

                input_data = []
                for fname in trained.features:
                    input_data.append(form.cleaned_data[fname])

                import pandas as pd
                X_new = pd.DataFrame([input_data], columns=trained.features)

                if scaler is not None:
                    X_new_scaled = scaler.transform(X_new)
                else:
                    X_new_scaled = X_new

                prediction = model.predict(X_new_scaled)[0]

                if trained.model_type == 'linear':
                    return render(request, 'classifier/predict.html', {
                        'model': trained,
                        'form': form,
                        'dataset': trained.dataset,
                        'prediction': round(float(prediction), 4),
                        'input_values': input_data,
                        'feature_names': trained.features,
                    })

                prediction_class = trained.classes[int(prediction)]

                probabilities = model.predict_proba(X_new_scaled)[0]
                prob_dict = {}
                for i, cls_name in enumerate(trained.classes):
                    prob_dict[cls_name] = round(float(probabilities[i]) * 100, 2)

                return render(request, 'classifier/predict.html', {
                    'model': trained,
                    'form': form,
                    'dataset': trained.dataset,
                    'prediction': prediction_class,
                    'probabilities': prob_dict,
                    'input_values': input_data,
                    'feature_names': trained.features,
                })
            except Exception as e:
                messages.error(request, f'Error al realizar la predicción: {str(e)}')
                return render(request, 'classifier/predict.html', {
                    'model': trained,
                    'form': form,
                    'dataset': trained.dataset,
                })
    else:
        form = PredictForm(features=features)

    return render(request, 'classifier/predict.html', {
        'model': trained,
        'form': form,
        'dataset': trained.dataset,
    })


def delete_dataset(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    models = TrainedModel.objects.filter(dataset=dataset)
    for m in models:
        if m.model_file:
            try:
                os.remove(m.model_file.path)
            except Exception:
                pass
        m.delete()
    try:
        os.remove(dataset.file.path)
    except Exception:
        pass
    dataset.delete()
    messages.success(request, 'Dataset eliminado correctamente.')
    return redirect('datasets')


def delete_model(request, pk):
    trained = get_object_or_404(TrainedModel, pk=pk)
    pk_val = trained.dataset.pk
    if trained.model_file:
        try:
            os.remove(trained.model_file.path)
        except Exception:
            pass
    trained.delete()
    messages.success(request, 'Modelo eliminado correctamente.')
    return redirect('dataset_detail', pk=pk_val)


def tutorial(request):
    regularization_plot = utils.generate_regularization_plot()
    return render(request, 'classifier/tutorial.html', {
        'regularization_plot': regularization_plot,
    })
