from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.Main,name='Main' ),
    path('Register',views.Register,name='Register'),
    path('Login',views.Login,name='Login'),
    path('Mask',views.Mask,name='Mask'),
    path('Maskpic',views.Maskpic,name='Maskpic'),
    path('Maskchoice',views.Maskchoice,name='Maskchoice'),
    path('video',views.video,name='video'),
    path('Removal',views.Removal,name='Removal'),
    path('activate/<uidb64>/<token>',views.activate,name='activate'),
    path('clear-temporary-files/', views.clear_temporary_files, name='clear_temporary_files'),
]