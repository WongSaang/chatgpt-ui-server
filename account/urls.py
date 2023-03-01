from django.urls import path, include
from . import views
from allauth.account.views import confirm_email

urlpatterns = [
    path('signup', views.signup, name='signup'),
    path('/', include('allauth.urls')),
    path('confirm-email/<str:key>/', confirm_email, name='account_confirm_email'),
]
