from django.urls import path,include
import legal.views as views

urlpatterns = [
    
     path('document-creation',views.DocumentCreation.as_view(),name='document-creation'),
      path('gold-contract',views.GoldContract.as_view(),name='gold-contract'),
]