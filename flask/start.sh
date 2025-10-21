#!/bin/bash
export MODEL_PATH=/opt/my-flask-app/model.joblib
exec gunicorn -w 4 -b 0.0.0.0:8080 app:app
