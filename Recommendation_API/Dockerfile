# syntax=docker/dockerfile:1

FROM python:3.6

COPY ./data ./ \
    ./herp ./ \
    ./result ./ \
    ./main.py ./ \
    ./files/wsgi.py ./ \
    ./wsgi.py ./ \
    ./requirements.txt ./

RUN apt-get update

RUN python3.6 -m pip install --upgrade pip && pip install -r requirements.txt

RUN apt-get install -y vim
    
RUN apt-get install -y nginx 

# Remove the default config 
RUN rm /etc/nginx/sites-available/default

COPY ./default /etc/nginx/sites-available/default

# container port
EXPOSE 80

# CMD ["python", "ML_API.py"]