from django.contrib import admin
from django.urls import path
import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('analysis/', views.analysis, name='analysis'),
    path('predict/', views.predict, name='predict'),
]