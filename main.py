import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Model Prediction with Tensorflow 
def model_prediction(test_image):
    model_path = r"C:/Users/user/Desktop/Projectdataset/corn_disease_detection_model.keras"
    
    if os.path.exists(model_path):
        try:
            # Loading the model with custom objects if any
            model = tf.keras.models.load_model(model_path)
            
            # Preprocessing the images for prediction
            image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to batch
            predictions = model.predict(input_arr)
            
            return np.argmax(predictions)  # Return index of max element
        except Exception as e:
            st.error(f"Error loading or using the model: {e}")
            return None
    else:
        st.error("Model file not found at the specified path.")
        return None

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        background-size: cover;
    }
    .header-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    .header-logo img {
        width: 150px;
    }
    .main-content {
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        margin-top: 20px;
    }
    .nav-tabs {
        display: flex;
        justify-content: space-around;
        background-color: #28a745;
        padding: 10px 0;
        border-radius: 5px;
    }
    .nav-tabs button {
        background: none;
        border: none;
        color: white;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        padding: 10px 20px;
        transition: background 0.3s;
    }
    .nav-tabs button:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    .nav-tabs .active {
        background-color: white;
        color: #28a745;
        border-radius: 5px;
    }
    .footer {
        padding: 20px;
        text-align: center;
        font-size: 14px;
        color: white;
        margin-top: 20px;
        background-color: #28a745;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define tabs
tabs = ["Home", "About", "Disease Recognition"]
active_tab = st.selectbox("Navigation", tabs, key="navigation")

# Header
st.markdown(
    """
    <div class='header-logo'>
        <img src='C:/Users/user/Desktop/Projectdataset/Logo.jpeg'>
    </div>
    """,
    unsafe_allow_html=True
)

# Home Page
if active_tab == "Home":
    st.header("CORN AI LEAF DISEASE RECOGNITION SYSTEM")
    st.image("Logo.jpeg", use_column_width=True)
    st.markdown(
        """
        <div class="main-content">
        Welcome to **Corn_AI** Leaf Disease Recognition System! 🌽🔍

        Our mission is to efficiently identify leaf diseases in maize. 
        You just have to upload an image of a plant, and our system will analyze it to detect any signs of disease. 
        Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Visit the **Disease Recognition** page and upload a photo of a plant with suspected diseases.
        2. **Analysis:** Our system processes the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and get recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** We use state-of-the-art machine learning techniques for precise disease detection.
        - **User-Friendly:** Our interface is simple and intuitive for a seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, enabling quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** page.
        </div>
        """, 
        unsafe_allow_html=True
    )

# About Project
elif active_tab == "About":
    st.header("About")
    st.markdown(
        """
        <div class="main-content">
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset was uploaded by Plant Village fromwhere we derived the data related to Corn.
        This dataset consists of 7316 rgb images of healthy and diseased corn leaves which is categorized into 4 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 3 test images is created later for prediction purpose.

        </div>
        """, 
        unsafe_allow_html=True
    )

# Prediction Page
elif active_tab == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=400, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    # Define your class names based on your model
                    class_names = [
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust', 
                        'Corn_(maize)___Northern_Leaf_Blight', 
                        'Corn_(maize)___healthy'
                    ]
                    st.success(f"Model predicts it's a {class_names[result_index]}")
                else:
                    st.error("Could not make a prediction. Please try again.")

# Footer
st.markdown(
    """
    <div class="footer">
        © 2024 Corn AI. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
