from django.contrib import admin


from .models import TokenUsage


@admin.register(TokenUsage)
class TokenUsageAdmin(admin.ModelAdmin):
    list_display = ('user', 'tokens')
