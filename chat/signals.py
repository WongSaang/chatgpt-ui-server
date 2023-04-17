from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import Setting

@receiver(post_migrate)
def load_default_settings(sender, **kwargs):
    if sender.name == 'chat':
        print('Setting up default settings...')
        if not Setting.objects.filter(name='open_frugal_mode_control').exists():
            Setting.objects.create(name='open_frugal_mode_control', value='True')
            print('Created setting: open_frugal_mode_control')
        if not Setting.objects.filter(name='open_registration').exists():
            Setting.objects.create(name='open_registration', value='True')
            print('Created setting: open_registration')
        if not Setting.objects.filter(name='open_web_search').exists():
            Setting.objects.create(name='open_web_search', value='False')
            print('Created setting: open_web_search')
        if not Setting.objects.filter(name='open_api_key_setting').exists():
            Setting.objects.create(name='open_api_key_setting', value='False')
            print('Created setting: open_api_key_setting')