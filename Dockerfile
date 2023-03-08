FROM python:3.10-slim as wsgi-server

RUN apt update \
    && apt install -y --no-install-recommends python3-dev default-libmysqlclient-dev build-essential libpq-dev dos2unix \
    && rm -rf /var/lib/apt/lists/*

ENV DJANGO_SUPERUSER_USERNAME=admin
ENV DJANGO_SUPERUSER_PASSWORD=password
ENV DJANGO_SUPERUSER_EMAIL=admin@example.com

COPY requirements.txt ./

RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt

WORKDIR /app

COPY . .

RUN python manage.py check --deploy \
    && python manage.py collectstatic --no-input \
    && dos2unix entrypoint.sh \
    && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

EXPOSE 8000


FROM nginx:1.22-alpine as web-server

WORKDIR /app

COPY --from=wsgi-server /app/static /app/static

COPY nginx.conf /etc/nginx/templates/default.conf.template
