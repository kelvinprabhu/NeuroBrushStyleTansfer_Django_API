from django.urls import path
from .views import demo_api
from .views import style_transfer_api
urlpatterns = [
    path('demo/', demo_api, name='demo_api'),
    path('style-transfer/', style_transfer_api, name='style_transfer_api'),
]
