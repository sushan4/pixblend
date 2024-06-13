import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_faces(user_image):
    new_img = np.array(user_image.convert("RGB"))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 6)
    for (x, y, w, h ) in faces:
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, faces

def detect_eyes(user_image):
    new_img = np.array(user_image.convert("RGB"))
    eyes=eye_cascade.detectMultiScale(new_img, 1.3, 5)
    for(x,y,w,h) in eyes:
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (0,255,0), 2)
    return new_img

def cartoonize_image(user_image):
    new_img = np.array(user_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(new_img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color,  mask = edges)

    return cartoon

def cannize_image(user_image):
    new_img = np.array(user_image.convert("RGB"))
    img = cv2.GaussianBlur(new_img, (13, 13), 0)
    canny = cv2.Canny(img, 80, 120)
    return canny




def main():
    st.title('PixBlend - Image Editor')
    st.text('Python Powered, Web based Image Editor')


    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            user_image = Image.open(image_file)
            st.text('Original Image')
            st.image(user_image)

            enhance_type = st.sidebar.radio("Enhance type", ['Original', 'Grayscale', 'Contrast', 'Brightness', 'Blur',
                                                             'Sharpness'])
            if enhance_type == "Grayscale":
                img = np.array(user_image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray)
            elif enhance_type == 'Contrast':
                rate = st.sidebar.slider("Contrast Level", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(user_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == 'Brightness':
                rate = st.sidebar.slider("Brightness Level", 0.0, 8.0)
                enhancer = ImageEnhance.Brightness(user_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type=='Blur':
                rate = st.sidebar.slider("Blur Level", 0.0, 7.0)
                blur_img = cv2.GaussianBlur(np.array(user_image), (15,15), rate)
                st.image(blur_img)
            elif enhance_type == 'Sharpness':
                rate = st.sidebar.slider("Sharpness Level", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(user_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == 'Original':
                st.image(user_image)
            else:
                st.image(user_image)
        tasks=['Faces', 'Eyes', 'Cartoonize', 'Cannize']
        feature_choice = st.sidebar.selectbox('Find features', tasks)
        if st.button("Process"):
            if feature_choice == "Faces":
                result_img, result_face = detect_faces(user_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_face)))
            elif feature_choice == "Eyes":
                result_img = detect_eyes(user_image)
                st.image(result_img)
            elif feature_choice == "Cartoonize":
                result_img = cartoonize_image(user_image)
                st.image(result_img)
            elif feature_choice == "Cannize":
                result_img = cannize_image(user_image)
                st.image(result_img)





    elif choice == 'About':
        st.subheader('About the Developer')
        st.markdown('Built with ❤️ by [Sushan Uchil](https://www.sushanuchil.tech/)')
        st.markdown('My LinkedIn profile - [Linkedin](https://www.linkedin.com/in/sushanuchil/)')





if __name__=='__main__':
    main()