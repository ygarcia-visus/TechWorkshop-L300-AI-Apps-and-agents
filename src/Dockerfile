# syntax=docker/dockerfile:1.4
# Example Dockerfile that copies only the contents of the src/ folder (build context is ./src)
# Adjust base image and commands as needed for your application.
FROM python:3.11-slim

WORKDIR /app

# Copy application sources (context is ./src so only those files are included)
COPY . /app

# Install Python dependencies if provided
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Entrypoint writes the ENV env var to /app/.env at container start (if provided) and then runs CMD
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Replace with your application's default command
CMD ["python", "app.py"]