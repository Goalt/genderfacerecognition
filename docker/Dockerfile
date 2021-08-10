FROM jasonrandrews/pytorch-dev

RUN mkdir models
RUN mkdir plots

RUN sudo apt-get update
RUN sudo apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

RUN pip install streamlit==0.71.0
RUN pip install Pillow==8.0.1
RUN pip install opencv-python==4.4.0.46

COPY ./plots/ plots/
COPY ./src/site.py site.py
COPY ./src/model.py model.py
COPY ./models/haarcascade_frontalface_default.xml models/haarcascade_frontalface_default.xml
COPY ./models/gfcr_v4.pt models/gfcr_v4.pt

RUN echo ~/.streamlit/config.toml >> enableCORS = false

CMD streamlit run site.py --server.maxUploadSize=10