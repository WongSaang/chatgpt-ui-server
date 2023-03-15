from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import Setting

@receiver(post_migrate)
def load_default_settings(sender, **kwargs):
    if sender.name == 'chat':
        print('Setting up default settings...')
        if not Setting.objects.filter(name='openai_api_key').exists():
            Setting.objects.create(name='openai_api_key')
            print('Created setting: openai_api_key')
        if not Setting.objects.filter(name='open_registration').exists():
            Setting.objects.create(name='open_registration', value='True')
            print('Created setting: open_registration')