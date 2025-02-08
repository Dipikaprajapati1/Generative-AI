import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from langchain_groq import ChatGroq 
import tensorflow as tf
from langchain.schema import HumanMessage
from PIL import Image
import os

def capture_photo():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        st.error("Error: Camera not accessible.")
        return None

    ret, frame = cap.read()
    cap.release() 

    if ret:
        img_path = "live_photo.jpg"
        cv2.imwrite(img_path, frame)
        return img_path
    else:
        st.error("Failed to capture image.")
        return None

def verify_faces(photo1, photo2):
    try:
        result = DeepFace.verify(photo1, photo2, model_name="Facenet", enforce_detection=False)
        similarity = 1 - result["distance"] 
        return result["verified"], similarity
    except Exception as e:
        st.error("Face verification failed: " + str(e))
        return False, 0

def get_verification_message(is_verified, similarity_score):
    chat = ChatGroq(model="mixtral-8x7b-32768", api_key="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")

    if is_verified:
        prompt = f"The user verification was successful. The similarity score is {similarity_score:.2f}. Generate a friendly confirmation message."
    else:
        prompt = f"Verification failed. Similarity score: {similarity_score:.2f}. Explain why and suggest re-verification."

    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

st.title("🔍 KYC Face Verification System")

id_photo = st.file_uploader("📸 Upload your ID photo (e.g., Aadhaar Card)", type=["jpg", "png", "jpeg","jfif"])

if id_photo:
    id_photo_path = "id_photo.jpg"
    with open(id_photo_path, "wb") as f:
        f.write(id_photo.read())

    st.image(id_photo_path, caption="Uploaded ID Photo", use_column_width=True)

    if st.button("📷 Capture Live Photo"):
        live_photo_path = capture_photo()
        if live_photo_path:
            st.image(live_photo_path, caption="Captured Live Photo", use_column_width=True)

            is_verified, similarity_score = verify_faces(id_photo_path, live_photo_path)

            verification_message = get_verification_message(is_verified, similarity_score)
            st.write("📝 **Verification Result:**", verification_message)
            
            if is_verified:
                st.success("✅ KYC Verified!")
            else:
                st.warning("❌ Verification Failed! Please try again.")