from django.db import models
from rest_framework import serializers
from .models import Conversation, Message, Prompt, EmbeddingDocument, Setting

class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ['id', 'topic', 'created_at']

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'message', 'is_bot', 'message_type', 'embedding_message_doc', 'created_at']


class PromptSerializer(serializers.ModelSerializer):

    prompt = serializers.CharField(trim_whitespace=False, allow_blank=True)

    class Meta:
        model = Prompt
        fields = ['id', 'title', 'prompt', 'created_at', 'updated_at']


class EmbeddingDocumentSerializer(serializers.ModelSerializer):
    '''embedding document store'''
    class Meta:
        ''' select fields'''
        model = EmbeddingDocument
        fields = ['id', 'title', 'created_at']
        read_only_fields = ('faiss_store', 'created_at')


class SettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Setting
        fields = ('name', 'value')