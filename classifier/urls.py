from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('datasets/', views.datasets_list, name='datasets'),
    path('upload/', views.upload_dataset, name='upload'),
    path('dataset/<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('dataset/<int:pk>/train/', views.train_model, name='train'),
    path('models/', views.models_list, name='models'),
    path('model/<int:pk>/', views.model_detail, name='model_detail'),
    path('model/<int:pk>/evaluate/', views.evaluate_model, name='evaluate'),
    path('model/<int:pk>/predict/', views.predict_view, name='predict'),
    path('delete_dataset/<int:pk>/', views.delete_dataset, name='delete_dataset'),
    path('delete_model/<int:pk>/', views.delete_model, name='delete_model'),
    path('tutorial/', views.tutorial, name='tutorial'),
]
