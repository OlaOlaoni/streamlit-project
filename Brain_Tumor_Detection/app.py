import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification

st.title("Image Classification with Google's Teachable Machine")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

uploaded_file = st.sidebar.file_uploader("Choose a brain MRI Image ...", type=['jpeg','jpg','png'])
btn = st.button('Button')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(uploaded_file, width=240)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)

    if btn:
        if image:
            st.write("")
            st.write("Classifying...")
            label = teachable_machine_classification(image, 'keras_model.h5')
            if label == 0:
                st.write("The MRI scan has a brain tumor")
            elif label == 1:
                st.write("The MRI scan is healthy")
            else:
                raise TypeError()
        elif not image:
            st.text("No Image selected")
            st.warning('Please input a name.')
