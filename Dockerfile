FROM python:3.9-slim

# Copy all files to the /app directory
COPY . /app
WORKDIR /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Run the Streamlit app on port 8080 and listen on all interfaces
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
