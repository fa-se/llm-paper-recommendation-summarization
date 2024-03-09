# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install git (so that we can install python packages from github via pip)
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt