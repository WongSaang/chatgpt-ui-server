from django.contrib import admin

from .models import Conversation, Message, Setting

admin.site.register(Conversation)
admin.site.register(Message)

class SettingAdmin(admin.ModelAdmin):
    list_display = ('name', 'value')

admin.site.register(Setting, SettingAdmin)