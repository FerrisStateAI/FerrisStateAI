FROM python:3.11

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the app and templates folder into the container
COPY ./app /code/app
COPY ./templates /code/templates

# Expose the port on which the app will run
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]