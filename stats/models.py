from django.db import models
from django.contrib.auth.models import User


class TokenUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    tokens = models.IntegerField(default=0)
