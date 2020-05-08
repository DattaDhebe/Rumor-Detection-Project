from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('home/data/', views.data, name='data'),
    path('home/data/graph/', views.graph, name='graph'),
    path('home/data/result/', views.result, name='result')

]
