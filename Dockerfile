# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app

COPY data/ app/data/
COPY eval.dvc /app/eval.dvc
COPY kaggle_prediction.dvc /app/kaggle_prediction.dvc
COPY package /app/package
COPY requirements.txt /app/requirements.txt
COPY split_train.dvc /app/split_train.dvc
COPY train_model.dvc /app/train_model.dvc
#COPY .dvc /app/.dvc
COPY go.sh /app/go.sh

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git  &&  \
    pip install --trusted-host pypi.python.org -r requirements.txt &&  \
    pip install dvc[all] &&  \
    dvc init

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
#CMD ["python", "app.py"]
#CMD ["dvc", "repro", "eval.dvc"]
#CMD ./go.sh
