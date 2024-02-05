"""
URL configuration for ds_nlp_models project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
import ds_nlp_models_app.views as views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('topicmodelling',views.SummerizeModel.as_view(),name='summerize'),
    path('churnPredict',views.ChurnPredictionModel.as_view(),name='churnPredict'),
    path('getFile',views.GetChurnPredictionOutputFile.as_view(),name='getFile'),
     path('translate',views.TranslateModel.as_view(),name='translate'),
       path('pdf-summary',views.SummaryModel.as_view(),name='pdf-summary'),
        path('actuator/health',views.HealthCheckModel.as_view(),name='actuator/health')
      
]
