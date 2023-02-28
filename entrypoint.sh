#!/bin/bash

python manage.py migrate

python manage.py createsuperuser --no-input

exec uvicorn chatgpt_ui_server.asgi:application --proxy-headers --host 0.0.0.0 --port 8000