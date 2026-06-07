# Regresión Lineal y Logística

Aplicación web Django para aprendizaje supervisado con regresión logística (clasificación) y regresión lineal (predicción numérica). Interfaz en español con Bootstrap 5 (tema oscuro). Incluye visualizaciones educativas, escalado automático y datasets precargados.

🚀 **Demo en vivo:** [andresbravo.pythonanywhere.com](https://andresbravo.pythonanywhere.com)

## Características

- **Dos tipos de modelo**: Regresión Logística (clasificación binaria y multiclase) y Regresión Lineal (predicción numérica continua).
- **Datasets precargados**: Iris (150 muestras, 3 clases), Cáncer de Mama (569 muestras, binario), Diabetes (442 muestras, regresión).
- **Subida de datasets**: Carga archivos CSV propios con selección de delimitador (coma o punto y coma).
- **Entrenamiento configurable**: Parámetros ajustables (test_size, C de regularización, max_iter, random_state).
- **Evaluación completa**:
  - *Logística*: Accuracy, precisión/recall/F1 por clase, matriz de confusión, curva ROC + AUC.
  - *Lineal*: R², RMSE, MSE, MAE, gráfico de residuales, valores reales vs predichos.
- **Predicción**: Formulario dinámico generado a partir de las características del modelo.
- **Visualizaciones educativas**: Función sigmoide, efecto de regularización (C), matriz de confusión explicada, coeficientes como barras CSS interactivas.
- **Escalado automático**: StandardScaler aplicado antes del entrenamiento; los coeficientes se interpretan en unidades de desviación estándar.
- **Persistencia**: Modelo + scaler guardados juntos en formato joblib.
- **Tutorial interactivo**: Guía paso a paso con explicaciones de cada funcionalidad, tabla comparativa de modelos, acordeón de parámetros, pestañas de métricas y FAQ integrado.

## Tecnologías

| Componente | Versión |
|---|---|
| Python | 3.12 |
| Django | 6.0.6 |
| scikit-learn | 1.9.0 |
| pandas | 2.0+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| joblib | 1.2+ |
| Bootstrap | 5 (tema oscuro) |
| Base de datos | SQLite |

## Demo en vivo

Puedes probar la aplicación sin instalar nada:

👉 [https://andresbravo.pythonanywhere.com](https://andresbravo.pythonanywhere.com)

Incluye los tres datasets precargados. Los datos subidos y modelos entrenados son persistentes.

## Inicio rápido

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

## Rutas y vistas

| Ruta | Vista | Descripción |
|---|---|---|
| `/` | `index` | Página principal con estadísticas y gráficos educativos |
| `/datasets/` | `datasets_list` | Lista de todos los datasets disponibles |
| `/upload/` | `upload_dataset` | Subir nuevo dataset en formato CSV |
| `/dataset/<pk>/` | `dataset_detail` | Vista previa (10 filas), tipos de columna, estadísticas |
| `/dataset/<pk>/train/` | `train_model` | Configurar y entrenar un modelo |
| `/models/` | `models_list` | Lista de todos los modelos entrenados |
| `/model/<pk>/` | `model_detail` | Coeficientes, interpretación, gráficos educativos |
| `/model/<pk>/evaluate/` | `evaluate_model` | Métricas de evaluación y gráficos |
| `/model/<pk>/predict/` | `predict_view` | Formulario para predecir con el modelo |
| `/tutorial/` | `tutorial` | Tutorial interactivo con guías, parámetros, métricas y FAQ |
| `/delete_dataset/<pk>/` | `delete_dataset` | Eliminar dataset y sus modelos asociados |
| `/delete_model/<pk>/` | `delete_model` | Eliminar un modelo entrenado |

## Modelo de datos

### Dataset

| Campo | Tipo | Descripción |
|---|---|---|
| `name` | CharField | Nombre del dataset |
| `file` | FileField | Archivo CSV |
| `target_column` | CharField | Columna objetivo seleccionada |
| `description` | TextField | Descripción opcional |
| `is_preloaded` | BooleanField | Indica si es un dataset precargado |
| `rows` | IntegerField | Número de filas |
| `columns` | IntegerField | Número de columnas |
| `created_at` | DateTimeField | Fecha de creación |

### TrainedModel

| Campo | Tipo | Descripción |
|---|---|---|
| `dataset` | ForeignKey | Dataset asociado |
| `name` | CharField | Nombre del modelo |
| `model_file` | FileField | Archivo joblib del modelo + scaler |
| `features` | JSONField | Lista de características usadas |
| `target` | CharField | Nombre de la columna objetivo |
| `coefficients` | JSONField | Coeficientes del modelo por clase |
| `intercept` | FloatField | Intercepto (β₀) |
| `metrics` | JSONField | Métricas de evaluación |
| `test_size` | FloatField | Proporción de datos de prueba |
| `params` | JSONField | Parámetros de entrenamiento |
| `classes` | JSONField | Lista de clases (solo logística) |
| `is_binary` | BooleanField | Indica si es clasificación binaria |
| `model_type` | CharField | `logistic` o `linear` |
| `created_at` | DateTimeField | Fecha de creación |

## Flujo de trabajo

```
1. Seleccionar dataset ─────────────────────────┐
   │ Pre-cargado (Iris, Cáncer, Diabetes)       │
   └ Subir CSV propio                           │
                                                ▼
2. Elegir tipo de modelo ───────────────────────┐
   │ Regresión Logística (clasificación)        │
   └ Regresión Lineal (predicción numérica)     │
                                                ▼
3. Configurar parámetros ───────────────────────┐
   │ Columna objetivo                           │
   │ Proporción de prueba (test_size)           │
   │ C (regularización, solo logística)         │
   │ Iteraciones máximas (solo logística)       │
   └ Semilla aleatoria                          │
                                                ▼
4. Entrenar modelo ─────────────────────────────┤
                                                ▼
5. Evaluar ─────────────────────────────────────┐
   │ Logística: accuracy, F1, matriz, ROC       │
   └ Lineal: R², RMSE, residuales, real vs pred │
                                                ▼
6. Predecir ────────────────────────────────────┐
   └ Ingresar valores → obtener predicción      │
```

## Formato de datos esperado

- Archivo en formato **CSV** con cabecera en la primera fila.
- Delimitador: **coma (`,`)** o **punto y coma (`;`)**.
- Al menos una columna debe ser la **variable objetivo**.
- Las columnas numéricas se usan como **características**.
- Para **clasificación**: la variable objetivo debe tener entre 2 y 20 valores únicos.
- Para **regresión**: la variable objetivo debe ser numérica continua con al menos 5 valores distintos.
- Las columnas de texto se codifican automáticamente (one-hot encoding).
- Los valores faltantes se rellenan con la media de cada columna.

## Estructura del proyecto

```
regresion-lineal/
├── clasificador_app/           # Configuración Django
│   ├── settings.py             # Configuración general
│   ├── urls.py                 # Rutas raíz
│   └── wsgi.py                 # WSGI para producción
├── classifier/                 # App principal
│   ├── forms.py                # Formularios (upload, train, predict)
│   ├── models.py               # Modelos Dataset y TrainedModel
│   ├── utils.py                # Lógica de ML y generación de gráficos
│   ├── views.py                # Vistas (11 rutas)
│   ├── urls.py                 # Rutas de la app
│   ├── admin.py                # Configuración del admin
│   ├── management/
│   │   └── commands/
│   │       └── load_sample_datasets.py  # Carga datasets precargados
│   └── templates/classifier/   # Plantillas HTML
│       ├── base.html           # Layout base con Bootstrap 5 dark
│       ├── index.html          # Página principal
│       ├── datasets.html       # Lista de datasets
│       ├── upload.html         # Subir CSV
│       ├── dataset_detail.html # Detalle del dataset
│       ├── train.html          # Entrenar modelo
│       ├── tutorial.html       # Tutorial interactivo
│       ├── models.html         # Lista de modelos
│       ├── model_detail.html   # Detalle del modelo
│       ├── evaluate.html       # Evaluación
│       └── predict.html        # Predicción
├── datasets/                   # Datasets precargados (CSV)
├── media/                      # Archivos subidos (models, datasets)
├── static/                     # Archivos estáticos
├── staticfiles/                # Archivos estáticos recopilados (producción)
├── manage.py                   # CLI de Django
└── requirements.txt            # Dependencias
```

## Despliegue en PythonAnywhere

### 1. Clonar y configurar

```bash
git clone https://github.com/Andres-Developer-23/regresion-lineal.git
cd regresion-lineal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py load_sample_datasets
python manage.py collectstatic --noinput
```

### 2. Configurar Web App

Desde el panel de PythonAnywhere (pestaña Web):

- **Manual configuration** → Python 3.12
- **Source code:** `/home/tuusuario/regresion-lineal`
- **Working directory:** `/home/tuusuario/regresion-lineal`
- **Virtualenv:** `/home/tuusuario/regresion-lineal/venv`

### 3. WSGI file

Editar `/var/www/tuusuario_pythonanywhere_com_wsgi.py`:

```python
import os
import sys

path = '/home/tuusuario/regresion-lineal'
if path not in sys.path:
    sys.path.append(path)

os.environ['DJANGO_SETTINGS_MODULE'] = 'clasificador_app.settings'
os.environ['DJANGO_DEBUG'] = 'False'
os.environ['DJANGO_ALLOWED_HOSTS'] = 'tuusuario.pythonanywhere.com'

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

### 4. Static files

| URL | Directorio |
|---|---|
| `/static/` | `/home/tuusuario/regresion-lineal/staticfiles` |
| `/media/` | `/home/tuusuario/regresion-lineal/media` |

### 5. Recargar

Hacer clic en **Reload** y abrir `https://tuusuario.pythonanywhere.com`.

## Licencia

Proyecto educativo de código abierto.
