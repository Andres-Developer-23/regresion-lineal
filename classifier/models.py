import os
import uuid
from django.db import models


def dataset_upload_path(instance, filename):
    ext = filename.split('.')[-1]
    return f'datasets/{uuid.uuid4()}.{ext}'


def model_upload_path(instance, filename):
    return f'models/{uuid.uuid4()}.joblib'


class Dataset(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to=dataset_upload_path)
    target_column = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    is_preloaded = models.BooleanField(default=False)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def filename(self):
        return os.path.basename(self.file.name)


class TrainedModel(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    model_file = models.FileField(upload_to=model_upload_path, blank=True, null=True)
    features = models.JSONField()
    target = models.CharField(max_length=100)
    coefficients = models.JSONField(blank=True, null=True)
    intercept = models.FloatField(blank=True, null=True)
    metrics = models.JSONField(blank=True, null=True)
    feature_importance = models.JSONField(blank=True, null=True)
    test_size = models.FloatField(default=0.2)
    params = models.JSONField(blank=True, null=True)
    classes = models.JSONField(blank=True, null=True)
    is_binary = models.BooleanField(default=False)
    model_type = models.CharField(max_length=10, default='logistic')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.dataset.name})"
