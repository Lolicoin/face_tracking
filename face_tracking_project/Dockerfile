
FROM python:3.10-slim

# Installing dependencies for dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev

# Installing Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Installing dlib
RUN pip install dlib

# Setting the Qt platform to xcb to avoid the Wayland issue
ENV QT_QPA_PLATFORM=xcb

# Copying the source code
COPY src/ /app/src/

WORKDIR /app/src/

CMD ["python3", "main.py"]