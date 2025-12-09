#!/usr/bin/env bash
set -e

# If ENV environment variable is present, write it to /app/.env (or whichever path your app expects)
# Ensure the file is only readable by the container process user.
if [ -n "$ENV" ]; then
  printf '%s' "$ENV" > /app/.env
  chmod 600 /app/.env
fi

# Execute the container CMD
exec "$@"