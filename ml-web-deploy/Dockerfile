FROM python:3.13-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./scripts /code/scripts
COPY ./model /code/model
COPY ./static /code/static
COPY ./uploads /code/uploads

COPY ./index.html /code/index.html
COPY ./InstanceNormalization.py /code/InstanceNormalization.py
COPY ./main.py /code/main.py

CMD ["fastapi", "run", "main.py", "--port", "80"]
