# Use Python 3.11 base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir --upgrade setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (default Streamlit port)
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
