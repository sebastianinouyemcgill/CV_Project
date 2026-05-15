# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY app/ .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Grant access to webcam (for Linux/macOS)
ENV DISPLAY=:0

# Run the app
CMD ["python", "main.py"]