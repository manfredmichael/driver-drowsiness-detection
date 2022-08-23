FROM python:3.10-slim-buster
WORKDIR /app

COPY requirements.txt app/
RUN pip install -r app/requirements.txt

COPY app.py app/
COPY utils.py app/
COPY .env app/

ENTRYPOINT [ "streamlit" ]
CMD ["run", "app/app.py" ]

EXPOSE 8501
