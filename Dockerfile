FROM python:3.8.5-buster

WORKDIR /tmp
COPY requirements.txt /tmp

RUN pip install --upgrade pip --no-cache-dir && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    mkdir /tmp/app

WORKDIR /tmp/app
COPY ./project .