from django.db import models


class ApiKey(models.Model):
    key = models.CharField(max_length=100, unique=True)
    token_used = models.IntegerField(default=0)
    remark = models.CharField(max_length=255)
    is_enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.key
