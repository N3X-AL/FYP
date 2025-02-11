FROM python:3.11
COPY . /app
WORKDIR /app
RUN apt-get update && apt-get upgrade -y
RUN pip install -r Requirements.txt
CMD ["python3" , "newalgo.py", "-u"]
