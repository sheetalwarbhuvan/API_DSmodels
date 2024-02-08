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
from django.urls import path,include
import ds_nlp_models_app.views as views
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView


schema_view = get_schema_view(
    openapi.Info(
        title="AeriesOne",
        default_version='v1',),
    public=True,
    permission_classes=(permissions.AllowAny,),
    
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('topicmodelling',views.SummerizeModel.as_view(),name='summerize'),
    path('churnPredict',views.ChurnPredictionModel.as_view(),name='churnPredict'),
    path('getFile',views.GetChurnPredictionOutputFile.as_view(),name='getFile'),
    path('translate',views.TranslateModel.as_view(),name='translate'),
    path('pdf-summary',views.SummaryModel.as_view(),name='pdf-summary'),
     path('legal/',include("legal.urls")),
       
        path('actuator/health',views.HealthCheckModel.as_view(),name='actuator/health'),
        path('docs/', schema_view.with_ui('swagger', cache_timeout=0),name='schema-swagger-ui'),
        path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    # Optional UI:
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),


      
]
