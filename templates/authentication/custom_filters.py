import base64
from django import template

register = template.Library()

@register.filter
def bytes_to_base64(value):
    return base64.b64encode(value).decode('utf-8')