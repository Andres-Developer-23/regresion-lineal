# Regresión Lineal y Logística

Aplicación web Django para aprendizaje supervisado con regresión logística (clasificación) y regresión lineal (predicción numérica). Interfaz en español con Bootstrap 5 (tema oscuro).

## Características

- **Dos tipos de modelo**: Regresión Logística (clasificación) y Regresión Lineal (predicción numérica)
- **Datasets precargados**: Iris (clasificación multiclase), Cáncer de Mama (binaria), Diabetes (regresión)
- **Subida de datasets**: Carga archivos CSV propios
- **Entrenamiento**: Configuración de parámetros (test_size, C, max_iter, random_state)
- **Evaluación**: Accuracy, precisión/recall/F1, matriz de confusión, ROC/AUC (logística); R², RMSE, MSE, MAE, residuales (lineal)
- **Predicción**: Formulario dinámico por características
- **Visualizaciones educativas**: Sigmoide, regularización, matriz de confusión, coeficientes como barras CSS
- **Escalado automático**: StandardScaler aplicado antes del entrenamiento

## Tecnologías

- Python 3.12, Django 6.0.6, scikit-learn 1.9.0
- Bootstrap 5, SQLite
- matplotlib, seaborn, joblib, pandas, numpy

## Instalación

```bash
git clone https://github.com/Andres-Developer-23/regresion-lineal.git
cd regresion-lineal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py load_sample_datasets
python manage.py runserver
```

Abrir `http://localhost:8000` en el navegador.

## Estructura del proyecto

```
regresion-lineal/
├── clasificador_app/       # Configuración Django
│   └── settings.py
├── classifier/             # App principal
│   ├── forms.py            # Formularios
│   ├── models.py           # Dataset y TrainedModel
│   ├── utils.py            # Lógica de ML y gráficos
│   ├── views.py            # Vistas
│   ├── urls.py             # Rutas
│   └── templates/classifier/
├── datasets/               # Datasets precargados
├── media/                  # Archivos subidos
├── manage.py
└── requirements.txt
```

## Uso

1. Selecciona o sube un dataset
2. Elige tipo de modelo: **Regresión Logística** (clasificación) o **Regresión Lineal** (predicción numérica)
3. Configura parámetros y entrena
4. Evalúa el modelo con métricas y gráficos
5. Ingresa valores y predice
