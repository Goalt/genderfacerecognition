import streamlit as st
from model import FaceRecognitionPipeline, GenderRecognitionModelV2
import cv2
import torch
import numpy as np
from tempfile import NamedTemporaryFile


THRESHOLD = 0.41374868


@st.cache(hash_funcs={FaceRecognitionPipeline: lambda _: None, GenderRecognitionModelV2: lambda _: None})
def prepare(modelPath, haarConfFile, inputSize):
    model = torch.load(modelPath, map_location=torch.device('cpu') )
    gfcr = FaceRecognitionPipeline(model, haarConfFile, inputSize)
    return gfcr


if __name__ == '__main__':
    st.title('Gender Face Recognition')
    gfcr = prepare('./models/gfcr_v4.pt', './models/haarcascade_frontalface_default.xml', (48, 48))

    st.header('Gender Recognition with face detection')
    uploadedFileFirst = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
    tempFileFirst = NamedTemporaryFile(delete=True)
    if uploadedFileFirst is not None:
        # Convert the file to an opencv image.
        tempFileFirst.write(uploadedFileFirst.getvalue())
        tempFileFirst.close()
        opencvImage = cv2.imread(tempFileFirst.name)

        st.write('Original Image:')
        st.image(opencvImage, width=224, channels="BGR")

        res = gfcr.predict(opencvImage)

        if res[0] is None:
            st.write('No Faces Found')
        else:
            numberOfImages = res[0].shape[0]
            st.write('{} Faces Found'.format(res[0].shape[0]))
            
            inputImages = []
            caption = []
            for i in range(numberOfImages):
                inputImages.append(res[0][i][0].numpy())
                if (res[2][i].item() > THRESHOLD):
                    caption.append('Man {:.2f}%'.format(res[2][i].item() * 100))
                else:
                    caption.append('Woman {:.2f}%'.format(100 - res[2][i].item() * 100))

            st.image(inputImages, caption=caption, clamp=True, width=100)