FROM python:slim

RUN pip install mlflow databricks-cli

ENV MLFLOW_TRACKING_URI=databricks

#  Provide databricks host and token via environment variable
# export DATABRICKS_HOST="..."
# export DATABRICKS_TOKEN="..."

COPY models.sh /usr/bin
COPY models.py /usr/bin

COPY zscaler.pem /root/zscaler.pem

RUN cat /root/zscaler.pem >> /usr/local/lib/python3.8/site-packages/certifi/cacert.pem