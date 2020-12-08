import streamlit as st
from model import FaceRecognitionPipeline, GenderRecognitionModelV2
import cv2
import torch
import numpy as np


@st.cache(hash_funcs={FaceRecognitionPipeline: lambda _: None, GenderRecognitionModelV2: lambda _: None})
def prepare(modelPath, haarConfFile, inputSize):
    model = torch.load(modelPath)
    gfcr = FaceRecognitionPipeline(model, haarConfFile, inputSize)
    return gfcr


if __name__ == '__main__':
    st.title('Gender Face Recognition')
    gfcr = prepare('./models/gfcr_v4.pt', './models/haarcascade_frontalface_default.xml', (48, 48))


    uploadedFile = st.file_uploader('Upload image')
    if uploadedFile is not None:
        # Convert the file to an opencv image.
        fileBytes = np.asarray(bytearray(uploadedFile.read()), dtype=np.uint8)
        opencvImage = cv2.imdecode(fileBytes, 1)

        st.write('Original Image:')
        st.image(opencvImage, width=224, channels="BGR")

        res = gfcr.predict(opencvImage)

        if res[0] is None:
            st.write('No Faces Found')
        else:
            numberOfImages = res[0].shape[0]
            st.write('{} Faces Found, Threshold {}'.format(res[0].shape[0], 0.41374868))
            
            inputImages = []
            caption = []
            for i in range(numberOfImages):
                inputImages.append(res[0][i][0].numpy())
                caption.append('{:.3}'.format(res[2][i].item()))

            st.image(inputImages, caption=caption, clamp=True, width=100)
