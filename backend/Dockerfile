FROM python:3.9-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install\
	'ffmpeg'\
	'libsm6'\
	'libxext6'	-y

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .

EXPOSE 5000

CMD ["python", "backend.py"]  
