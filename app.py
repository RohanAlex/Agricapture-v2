import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras.models import load_model
import cv2
import firebase_admin
from firebase_admin import credentials, db
import os
import validators
from dotenv import load_dotenv
import altair as alt
import pandas as pd
import smtplib
from email.mime.text import MIMEText

load_dotenv()

FIREBASE_DB_URL = os.getenv('FIREBASE_DB_URL')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
RECEIVER = ["rohan.csb2024@saintgits.org" , "johanjp.csb2024@saintgits.org" , "ns.csb2024@saintgits.org" , "sneha.csb2024@saintgits.org"]
GMAIL_USERNAME = os.getenv('GMAIL_USERNAME')
body_flood = "Flood detected"
body_water = "Soil is dry!! Please Irrigate"
high_temp="Since the temperature is above 23.79°C there are high chances of diseases like Leaf Blast, Bacterial Leaf Blight and Brown Spot "

if not FIREBASE_DB_URL or not validators.url(FIREBASE_DB_URL):
    st.error("FIREBASE_DB_URL is missing or invalid. Please make sure to set a valid URL in your environment variables.")
    st.stop()

# Initialize Firebase Admin SDK
def initialize_firebase():
    if not firebase_admin._apps:
        # Initialize Firebase Admin SDK (replace 'serviceAccountKey.json' with your Firebase Admin SDK credentials file)
        cred = credentials.Certificate('credentials.json')
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
    return firebase_admin

st.set_page_config(
    page_title="AgriCapture",
    page_icon="agricaptureLogo.png",
    layout="wide"
)
st.sidebar.image("agricaptureLogo.png",use_column_width=True)



# Load the model and labels once during app startup
model = load_model("./model/rice_leaf_diseases.h5", compile=False)
class_names = [line.strip() for line in open("./model/diseases_labels.txt", "r")]

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Initialize Firebase outside of main
firebase_admin = initialize_firebase()

# Get a reference to the Firebase Realtime Database
db_ref = firebase_admin.db.reference('/')  # Replace with your Firebase Realtime Database path

# Load the CSV file
df = pd.read_csv("solutions.csv")

def get_categories():
    return df['Category'].unique()

def get_solutions(category):
    solutions_df = df[df['Category'] == category]
    return solutions_df[['Solution', 'Description', 'Dosage','Product Link']]


def preprocess_image(image):
    # Resize the image to be at least 224x224 and then crop from the center
    size = (224, 224)
    return ImageOps.fit(image, size, Image.Resampling.LANCZOS)

def is_leaf(image):
    # Convert the image to numpy array
    img_array = np.array(image)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define the lower and upper bounds of green color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Create a mask to threshold the image for green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to remove noise and fill gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Assuming the largest contour corresponds to the leaf
        contour_area = max([cv2.contourArea(contour) for contour in contours])
        # Adjust this threshold value based on the size of the leaf in your images
        if contour_area > 1000:  # Adjust threshold as needed
            return True

    return False

def predict_image(image):
    # Preprocess the image
    image = preprocess_image(image)

    # Check if it's a leaf
    if not is_leaf(image):
        st.error("Uploaded image does not contain a leaf.")
        return None, None

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score

def make_donut(input_response, input_text, input_color):
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=160, height=160) 
    
    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=22, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} '))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=160, height=160)
    return plot_bg + plot + text

def send_email(body):
    try:
        msg = MIMEText(body)
        msg['From'] = GMAIL_USERNAME
        msg['To'] = ', '.join(RECEIVER)
        msg['Subject'] = "Agricapture Alert"

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USERNAME,GMAIL_PASSWORD)
        server.sendmail(GMAIL_USERNAME,RECEIVER , msg.as_string())
        server.quit()

        print('Email sent successfully! 🚀')
    except Exception as e:
        print(f"Error in email : {e}")

def display_realtime_data():
    # Read the data from the Realtime Database
    data = db_ref.get()

    if data:
        st.write("Real-time Data:")
        keys = list(data.keys())
        cols = st.columns(3)
        for i, key in enumerate(keys):
            if key=='SoilMoisture':
                if data[key]==100:
                    send_email(body_flood)
                elif data[key]<=3:
                    send_email(body_water)
            if key=='Temperature':
                if data[key]>=23 :
                    send_email(high_temp)
                
                
                    pass
            with cols[i % 3]:
                st.text(key)
                st.altair_chart(make_donut(data[key], key, 'blue'), use_container_width=True)

    else:
        st.write("No data available in the Real-time Database.")

def home():
    st.title("AGRICAPTURE")
    st.write("""
        Welcome to the AGRICAPTURE. 
        This app helps in predicting diseases in rice leaves based on uploaded images or live camera capture.
        """)
    display_realtime_data()
    st.image('./paddy.png', caption="Paddy Field" )


def prediction():
    st.title("AGRICAPTURE")
    option = st.selectbox("Choose an option", ["Upload Image", "Take a picture"])

    if option == "Upload Image":
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Proceed"):
                st.write("Classifying...")
                class_name, confidence_score = predict_image(img)
                if class_name:
                    category = class_name[2:]
                    if category in get_categories():
                        st.write(f"Class: {class_name[2:]}")
                        st.write(f"Confidence Score: {confidence_score}")
                        solutions = get_solutions(category)
                        st.write("Solutions:")
                        st.table(solutions)
                    else:
                        st.write(f"No solutions found for category: {category}")

    elif option == "Take a picture":
        st.subheader("Live Camera Capture")

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer:
            img = Image.open(img_file_buffer)
            st.image(img, caption="Captured Image", use_column_width=True)

            if st.button("Proceed"):
                st.write("Classifying...")
                class_name, confidence_score = predict_image(img)
                if class_name:
                    category = class_name[2:]
                    if category in get_categories():
                        st.write(f"Class: {class_name[2:]}")
                        st.write(f"Confidence Score: {confidence_score}")
                        solutions = get_solutions(category)
                        st.write("Solutions:")
                        st.table(solutions)
                    else:
                        st.write(f"No solutions found for category: {category}")
                        

                        
                        
def main():
    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        home()
    elif choice == "Prediction":
        prediction()
    
        

if __name__ == "__main__":
    main()
