from django.contrib import admin
from django.forms import BooleanField
from django.forms.widgets import CheckboxInput
from .models import ApiKey


@admin.register(ApiKey)
class ApiKeyAdmin(admin.ModelAdmin):
    list_display = ('key', 'token_used', 'remark', 'is_enabled', 'created_at')

    formfield_overrides = {
        BooleanField: {'widget': CheckboxInput},
    }
