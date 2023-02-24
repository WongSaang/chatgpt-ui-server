from django.urls import include, path
from rest_framework import routers
from .views import ConversationViewSet, MessageViewSet

router = routers.SimpleRouter()
router.register(r'conversations', ConversationViewSet, basename='conversationModel')
router.register(r'messages', MessageViewSet, basename='messageModel')

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
]