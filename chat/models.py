import logging
from django.db import models
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)


class EmbeddingDocument(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    faiss_store = models.BinaryField(null=True)
    title = models.CharField(max_length=255, default="")
    created_at = models.DateTimeField(auto_now_add=True)


class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    messages = models.TextField(default='')
    tokens = models.IntegerField(default=0)
    is_bot = models.BooleanField(default=False)
    is_disabled = models.BooleanField(default=False)
    message_type = models.IntegerField(default=0)
    embedding_message_doc = models.ForeignKey(EmbeddingDocument, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    plain_message_type = 0
    hidden_message_type = 1
    temp_message_type = 2
    web_search_context_message_type = 100
    arxiv_context_message_type = 110
    doc_context_message_type  = 120

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        if self.message_type % 1000 > 100:  # remove document it attached
            pass
        super().delete(*args, **kwargs)


class Prompt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.TextField(null=True, blank=True)
    prompt = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class Setting(models.Model):
    name = models.CharField(max_length=255)
    value = models.CharField(max_length=255)
