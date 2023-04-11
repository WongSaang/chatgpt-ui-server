from rest_framework import serializers
from .models import Conversation, Message, Prompt, Setting

class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id', 'topic', 'created_at']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'message', 'is_bot', 'created_at']


class PromptSerializer(serializers.ModelSerializer):

    prompt = serializers.CharField(trim_whitespace=False, allow_blank=True)

    class Meta:
        model = Prompt
        fields = ['id', 'title', 'prompt', 'created_at', 'updated_at']


class SettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Setting
        fields = ('name', 'value')