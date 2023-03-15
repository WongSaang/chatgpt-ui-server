from rest_framework.response import Response
from rest_framework import status
from dj_rest_auth.registration.views import RegisterView
from chat.models import Setting


class RegistrationView(RegisterView):
    def create(self, request, *args, **kwargs):
        try:
            open_registration = Setting.objects.get(name='open_registration').value == 'True'
        except Setting.DoesNotExist:
            open_registration = True

        if open_registration:
            return super().create(request, *args, **kwargs)
        else:
            return Response({'detail': 'Registration is not yet open.'}, status=status.HTTP_403_FORBIDDEN)