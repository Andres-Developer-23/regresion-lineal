import os
from django.core.management.base import BaseCommand
from django.core.files import File
from classifier.models import Dataset


class Command(BaseCommand):
    help = 'Carga los datasets de ejemplo en la base de datos'

    def handle(self, *args, **options):
        samples = [
            {
                'name': 'Iris',
                'csv_path': 'datasets/iris.csv',
                'description': 'Dataset clásico de flores Iris. Contiene 150 muestras con 4 características '
                               '(largo/ancho de sépalo y pétalo) y 3 clases de flores (setosa, versicolor, virginica). '
                               'Problema de clasificación multiclase.',
            },
            {
                'name': 'Cáncer de Mama',
                'csv_path': 'datasets/cancer.csv',
                'description': 'Dataset de Cáncer de Mama (Wisconsin). Contiene 569 muestras con 30 características '
                               'numéricas derivadas de imágenes de aspiraciones con aguja fina. '
                               'Clasificación binaria: benigno vs maligno.',
            },
            {
                'name': 'Diabetes',
                'csv_path': 'datasets/diabetes.csv',
                'description': 'Dataset de Diabetes (sklearn). Contiene 442 muestras con 10 características '
                               'médicas (edad, sexo, IMC, presión, etc.) y un objetivo numérico '
                               'que mide la progresión de la diabetes un año después. '
                               'Problema de regresión lineal.',
            },
        ]

        for sample in samples:
            if Dataset.objects.filter(name=sample['name']).exists():
                self.stdout.write(f'  - "{sample["name"]}" ya existe, saltando.')
                continue

            csv_path = sample['csv_path']
            if not os.path.exists(csv_path):
                self.stdout.write(self.style.WARNING(f'  - Archivo {csv_path} no encontrado, saltando.'))
                continue

            with open(csv_path, 'rb') as f:
                dataset = Dataset(
                    name=sample['name'],
                    description=sample['description'],
                    is_preloaded=True,
                )
                dataset.file.save(os.path.basename(csv_path), File(f), save=False)

                import pandas as pd
                df = pd.read_csv(csv_path)
                dataset.rows = len(df)
                dataset.columns = len(df.columns)
                dataset.save()

            self.stdout.write(self.style.SUCCESS(f'  - "{sample["name"]}" cargado ({dataset.rows} filas, {dataset.columns} columnas).'))

        self.stdout.write(self.style.SUCCESS('Carga completada.'))
