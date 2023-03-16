from django.contrib import admin

from .models import Conversation, Message, Setting


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'topic', 'created_at')


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'get_conversation_topic', 'message', 'is_bot', 'created_at')

    def get_conversation_topic(self, obj):
        return obj.conversation.topic

    get_conversation_topic.short_description = 'Conversation Topic'


@admin.register(Setting)
class SettingAdmin(admin.ModelAdmin):
    list_display = ('name', 'value')