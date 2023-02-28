FROM python:3.10-slim as asgi-server

RUN apt update \
    && apt install -y --no-install-recommends python3-dev default-libmysqlclient-dev build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV DJANGO_SUPERUSER_USERNAME=admin
ENV DJANGO_SUPERUSER_PASSWORD=password
ENV DJANGO_SUPERUSER_EMAIL=admin@example.com

COPY requirements.txt ./

RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt

RUN groupadd -r appgroup && useradd -r -g appgroup appuser && mkdir -p /app && chown appuser /app

USER appuser

WORKDIR /app

COPY --chown=appuser . .

RUN python manage.py check --deploy \
    && mkdir static \
    && python manage.py collectstatic --no-input \
    && chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000


FROM nginx:1.22-alpine as web-server

WORKDIR /app

COPY --from=asgi-server /app/static /app/static

COPY nginx.conf /etc/nginx/templates/default.conf.template
