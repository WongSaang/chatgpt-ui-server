"""chatgpt_ui_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from chat.views import get_current_user, conversation, gen_title

urlpatterns = [
    path('api/auth/session', get_current_user, name='current_user'),
    path('api/auth/signin', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/token/refresh', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/chat/', include('chat.urls')),
    path('api/conversation', conversation, name='conversation'),
    path('api/gen_title', gen_title, name='gen_title'),
    path('admin/', admin.site.urls),
    path('api/account/', include('dj_rest_auth.urls')),
    path('api/account/registration/', include('dj_rest_auth.registration.urls'))
]
