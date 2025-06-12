# Gunakan Python 3.10 untuk kompatibilitas TensorFlow 2.13
FROM python:3.10-slim

# Instal dependencies dasar
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dan install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy semua source code
COPY . .

# Streamlit default port
EXPOSE 8501

# Jalankan aplikasi Streamlit
CMD ["streamlit", "run", "dashboard.py"]
