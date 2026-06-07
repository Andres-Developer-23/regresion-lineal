import pandas as pd
from django import forms


class DatasetUploadForm(forms.Form):
    name = forms.CharField(max_length=100, label='Nombre del dataset')
    file = forms.FileField(label='Archivo CSV')
    delimiter = forms.ChoiceField(
        choices=[(',', 'Coma (,)'), (';', 'Punto y coma (;)')],
        initial=',',
        label='Delimitador'
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        required=False,
        label='Descripción (opcional)'
    )

    def clean_file(self):
        file = self.cleaned_data['file']
        if not file.name.endswith('.csv'):
            raise forms.ValidationError('Solo se permiten archivos CSV.')
        return file


class TrainForm(forms.Form):
    regression_type = forms.ChoiceField(
        choices=[
            ('logistic', 'Regresión Logística (clasificación)'),
            ('linear', 'Regresión Lineal (predicción numérica)'),
        ],
        initial='logistic',
        label='Tipo de modelo',
        widget=forms.RadioSelect,
    )
    target_column = forms.ChoiceField(label='Columna objetivo (a predecir)')
    test_size = forms.FloatField(
        initial=0.2,
        min_value=0.05,
        max_value=0.5,
        label='Proporción de prueba',
        help_text='Entre 0.05 y 0.5'
    )
    c_param = forms.FloatField(
        initial=1.0,
        min_value=0.01,
        max_value=100.0,
        label='Parámetro C (inversa de regularización)',
        help_text='Valores pequeños = más regularización (solo regresión logística)',
        required=False,
    )
    max_iter = forms.IntegerField(
        initial=200,
        min_value=10,
        max_value=10000,
        label='Iteraciones máximas',
        required=False,
    )
    random_state = forms.IntegerField(
        initial=42,
        label='Semilla aleatoria'
    )

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', [])
        super().__init__(*args, **kwargs)
        if columns:
            self.fields['target_column'].choices = [(c, c) for c in columns]


class PredictForm(forms.Form):
    def __init__(self, *args, **kwargs):
        features = kwargs.pop('features', {})
        super().__init__(*args, **kwargs)
        for fname, ftype in features.items():
            self.fields[fname] = forms.FloatField(
                label=fname,
                required=True,
                widget=forms.NumberInput(attrs={'step': 'any', 'class': 'form-control'})
            )
