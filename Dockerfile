FROM python:3.10-alpine

RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories && apk update \
    && apk add --virtual build-deps gcc python3-dev musl-dev \
    && apk add --no-cache mariadb-dev nginx

ENV DJANGO_SUPERUSER_USERNAME=admin
ENV DJANGO_SUPERUSER_PASSWORD=password
ENV DJANGO_SUPERUSER_EMAIL=admin@example.com

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && apk del build-deps

COPY . .

RUN python manage.py check --deploy \
    && python manage.py collectstatic --no-input \
    && chmod +x entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

EXPOSE 8000