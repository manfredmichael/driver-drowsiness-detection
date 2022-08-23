FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt app/
RUN pip install -r app/requirements.txt

COPY background_subtraction.py app/
COPY app.py app/

# ENTRYPOINT [ "python" ]
# CMD ["app/app.py" ]

CMD gunicorn --bind 0.0.0.0:$PORT wsgi
