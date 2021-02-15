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
    
    '''
    ## Task

    | Parameter     | Value         |
    | ------------- |:-------------:|
    | Type | Binary Classification |
    | Input | Face Image 48 x 48 |
    | Metric | Accuracy |

    ## Dataset
    https://www.kaggle.com/nipunarora8/age-gender-and-ethnicity-face-data-csv

    | Parameter     | Value         |
    | ------------- |:-------------:|
    | Number of Images | 23.7k |
    | Image Size | 48 x 48 |
    | Test/train/val split |  20/70/10 |

    ## Project Pipeline
    The project's pipeline could be described in a such way:

    Streamlit -> OpenCV (Face detection) -> Pytorch (DNN) -> Streamlit

    ### OpenCV Face detection
    To detect and crop faces on image cv2.CascadeClassifier was used. Configure file was choosen "haarcascade_frontalface_default" (src.model.FaceRecognitionPipeline).

    After cv2.CascadeClassifier image was passed to CNN with sigmoid function as output.

    ### Pytorch DNN
    '''

    '''
    Train/val accuracy on each epoch:
    '''
    st.image("plots/acc.png", width=700)
    '''
    Train loss on each epoch:
    '''
    st.image("plots/train_loss.png", width=700)
    '''
    ROC curve:
    '''
    st.image("plots/roc_auc.png", width=700)
    '''
    DNN graph:
    '''
    st.image("plots/gfcr.png", width=300)

    '''
    ## Results
    | Parameter     | Value         |
    | ------------- |:-------------:|
    | Model | gfcr_v4.pt |
    | Train | 0.966 |
    | Val | 0.951 |
    | Test | 0.944 |

    ## Technologies

    Used packages:
    1. PyTorch
    2. OpenCV
    3. Pandas
    4. Streamlit
    5. Docker

    ## Usage Docker
    ```shell
    bash docker/docker_build.sh
    bash docker/docker_run.sh
    ```
    '''