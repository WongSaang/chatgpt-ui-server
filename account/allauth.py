from allauth.account.adapter import DefaultAccountAdapter
from allauth.utils import build_absolute_uri

class AccountAdapter(DefaultAccountAdapter):

    def get_email_confirmation_url(self, request, emailconfirmation):
        location = '/account/verify-email/{}'.format(emailconfirmation.key)
        return build_absolute_uri(None, location)