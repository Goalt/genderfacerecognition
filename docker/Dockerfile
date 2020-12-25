FROM ufoym/deepo:pytorch-py36-cpu

RUN mkdir models

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN pip install streamlit==0.72.0
RUN pip install Pillow==8.0.1
RUN pip install opencv-python==4.4.0.46

COPY ./src/site.py site.py
COPY ./src/model.py model.py
COPY ./models/haarcascade_frontalface_default.xml models/haarcascade_frontalface_default.xml
COPY ./models/gfcr_v4.pt models/gfcr_v4.pt

CMD streamlit run site.py