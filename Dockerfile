FROM python:3.8.5
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY app.py /app
COPY anomaly_detection_model.h5 /app
COPY scaler_data /app
COPY ohe_data /app
COPY ohe1_data /app
COPY qt_data /app
COPY qt1_data /app


#fastapi

CMD ["python", "anomaly_app.py"]
