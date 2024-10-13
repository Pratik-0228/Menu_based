from flask import Flask, request, render_template, jsonify , send_file
from twilio.rest import Client
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler, LabelEncoder
import google.generativeai as genai
import smtplib
import requests
import pyttsx3
import boto3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import subprocess
import subprocess as sp 
import shlex
import pandas as pd
import geocoder
from bs4 import BeautifulSoup
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes


app = Flask(__name__)

account_sid = os.getenv('')
auth_token = os.getenv('')


# Route for the homepage
@app.route('/')
def home():
    return render_template('index2.html')


#task1
# Function to perform Google Search
def google_search(query):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {'q': query, 'num': 10}
        response = requests.get('https://www.google.com/search', headers=headers, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results: {e}")
        return []

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []

        for item in soup.find_all('div', attrs={'class': 'g'}, limit=10):
            title_element = item.find('h3')
            link_element = item.find('a')
            snippet_element = item.find('span', attrs={'class': 'aCOpRe'})  # Updated to find the snippet correctly

            if title_element and link_element:
                title = title_element.text
                link = link_element['href']
                snippet = snippet_element.text if snippet_element else 'No snippet'
                results.append({
                    'title': title,
                    'link': link,
                    'snippet': snippet
                })

        print(f"Search results: {results}")  # Debug print
        return results
    return []

# Route for handling the search
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        results = google_search(query)
        return jsonify(results)
    return jsonify({"error": "No query provided"})



#task2
# fuction to send the Text msg
def send_text_msg(to_number, text):
    from twilio.rest import Client
    account_sid = ""
    auth_token = ""
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=text,
        from_="+16189480533",  # Your Twilio number
        to=to_number
    )
    return f"Message sent to {to_number}"

# Route to handle the POST request from the frontend to send the msg
@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    phone_number = data.get('phoneNumber')  # Get the phone number from the request
    text_message = data.get('text')

    if not phone_number or not text_message:
        return jsonify({"error": "No phone number or message provided"}), 400

    try:
        response = send_text_msg(phone_number, text_message)  # Pass the phone number to your function
        return jsonify({"success": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#task3
# function to find the geo location 
def get_current_location():
    try:
        # Get the current IP-based location
        g = geocoder.ip('me')
        if g.ok:
            location_data = {
                'latitude': g.latlng[0],
                'longitude': g.latlng[1],
                'city': g.city,
                'state': g.state,
                'country': g.country
            }
            return location_data
        else:
            return {"error": "Unable to get location"}
    except Exception as e:
        return {"error": str(e)}

# Route to get the current geolocation    
@app.route('/get_location', methods=['GET'])
def get_location():
    location_data = get_current_location()
    return jsonify(location_data)



#task4
# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "Speech synthesis complete."

# Route to handle the text-to-speech API call
@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    if 'text' in data:
        text = data['text']
        message = text_to_speech(text)
        return jsonify({'success': message})
    else:
        return jsonify({'error': 'No text provided'}), 400



#task5
# Function to set the system volume
def set_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

    # Convert volume level to the range expected by IAudioEndpointVolume
    volume_range = volume.GetVolumeRange()
    min_volume = volume_range[0]
    max_volume = volume_range[1]

    volume.SetMasterVolumeLevel(volume_level * (max_volume - min_volume) / 100 + min_volume, None)
    return "Volume set to {}%".format(volume_level)

# Route to control volume
@app.route('/set_volume', methods=['POST'])
def change_volume():
    data = request.json
    if 'volume' in data:
        volume_level = data['volume']
        if 0 <= volume_level <= 100:
            message = set_volume(volume_level)
            return jsonify({'success': message})
        else:
            return jsonify({'error': 'Volume level must be between 0 and 100'}), 400
    else:
        return jsonify({'error': 'No volume level provided'}), 400



#task 6
# Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
API_KEY = 'YOUR_API_KEY'

@app.route('/weather', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    if not city:
        return jsonify({"error": "City name is required"}), 400

    try:
        # Call OpenWeatherMap API
        response = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric')
        data = response.json()

        if response.status_code != 200:
            return jsonify({"error": data.get('message', 'Failed to fetch weather data')}), response.status_code

        # Extract necessary information
        temperature = data['main']['temp']
        condition = data['weather'][0]['description']

        return jsonify({
            "temperature": temperature,
            "condition": condition
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



#task7
# Function to send the email
def send_email(to_email, subject, body):
    from_email = ''
    from_password = ''
    
    # Create the email headers and body
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the Gmail server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Secure the connection
        server.login(from_email, from_password)  # Log in to the email account
        
        # Send the email
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return 'Email sent successfully!'
    except Exception as e:
        return f'Failed to send email. Error: {str(e)}'
    
# Flask route for sending email
@app.route('/send_email', methods=['POST'])
def send_email_route():
    data = request.json
    to_email = data.get('to_email')
    subject = data.get('subject')
    body = data.get('body')

    if not to_email or not subject or not body:
        return jsonify({"error": "Missing email fields"}), 400

    try:
        response = send_email(to_email, subject, body)
        return jsonify({"success": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



#aws
#task8
#function to launch ec2 instances
# Function to launch EC2 instance
# Route to handle launching EC2 instance
@app.route('/launch_ec2', methods=['POST'])
def launch_ec2():
    try:
        # Retrieve form data from the frontend
        instance_type = request.form['instance_type']
        ami_id = request.form['ami_id']
        key_name = request.form['key_name']
        security_group = request.form['security_group']
        
        # Initialize Boto3 EC2 client
        ec2 = boto3.client('ec2', region_name='ap-south-1')
        
        # Launch EC2 instance
        instance = ec2.run_instances(
            InstanceType=instance_type,
            ImageId=ami_id,
            KeyName=key_name,
            SecurityGroupIds=[security_group],
            MinCount=1,
            MaxCount=1
        )

        # Get the instance ID and status
        instance_id = instance['Instances'][0]['InstanceId']
        instance_state = instance['Instances'][0]['State']['Name']

        # Return success message and instance details
        return jsonify({
            'message': 'EC2 Instance launched successfully!',
            'instance_id': instance_id,
            'instance_state': instance_state
        })
    
    except Exception as e:
        # Handle any errors and return failure message
        return jsonify({'error': str(e)})


#task9
# Configure the API key
genai.configure(api_key="your api key")

# Define the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Function to generate response from prompt
def generate_response(prompt):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text

# API route to handle prompt input from frontend
@app.route('/generate', methods=['POST'])
def generate_prompt_response():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is missing'}), 400

    try:
        # Call the function to generate a response
        response_text = generate_response(prompt)
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



#task 10
#ML processed dataset
def load_dataset(file_path):
    """Load the dataset from the provided file path."""
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_data(data):
    """Clean the dataset by handling missing values."""
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    print("Missing values handled.")
    return data

def encode_categorical(data):
    """Encode categorical variables using Label Encoding."""
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
        print(f"Encoded column: {column}")
    return data, label_encoders

def scale_features(data):
    """Scale numeric features using Standard Scaler."""
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    print("Features scaled.")
    return data, scaler

#route to processed dataset
@app.route('/process', methods=['POST'])
def process_dataset():
    if 'file' not in request.files:
        return "No file uploaded.", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400
    
    # Save the file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Process the dataset
    data = load_dataset(file_path)
    if data is not None:
        data = clean_data(data)
        data, label_encoders = encode_categorical(data)
        data, scaler = scale_features(data)

        # Convert DataFrame to HTML table to display on frontend
        processed_data_html = data.to_html(classes='table table-striped', index=False)

        return render_template('result.html', processed_data_html=processed_data_html)
    else:
        return jsonify({"error": "Failed to process the dataset."}), 500
    
if not os.path.exists("uploads"):
    os.makedirs("uploads")   



#task 11
# Function to create the house image
def generate_house_image():
    # Define image dimensions
    height = 400
    width = 400

    # Create an empty image (3D array) of zeros - Black background
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Set the body of the house - A rectangle
    house_top_left_y, house_top_left_x = 250, 100
    house_bottom_right_y, house_bottom_right_x = 350, 300
    image[house_top_left_y:house_bottom_right_y, house_top_left_x:house_bottom_right_x] = [139, 69, 19]  # Brown house

    # Set the roof of the house - A triangle
    for y in range(150, 250):
        for x in range(100, 300):
            if abs(x - 200) <= (y - 150):  # Equation for the triangle shape
                image[y, x] = [255, 0, 0]  # Red roof

    # Set the door of the house - A smaller rectangle
    door_top_left_y, door_top_left_x = 300, 180
    door_bottom_right_y, door_bottom_right_x = 350, 220
    image[door_top_left_y:door_bottom_right_y, door_top_left_x:door_bottom_right_x] = [0, 0, 255]  # Blue door

    # Set the windows - Small squares
    # Left window
    image[270:300, 120:150] = [255, 255, 0]  # Yellow left window
    # Right window
    image[270:300, 250:280] = [255, 255, 0]  # Yellow right window

    # Save the image to a buffer
    buf = io.BytesIO()
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf

# Flask route to generate and send the house image
@app.route('/generate-house')
def generate_house():
    # Example of house image creation using NumPy and Pillow
    house_img = np.zeros((200, 200, 3), dtype=np.uint8)
    house_img[50:150, 50:150] = [255, 0, 0]  # Example: red square as house

    # Convert to image and serve it as response
    img = Image.fromarray(house_img)
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')



#task 12
# function to apply different filters on images 
def apply_filter_to_image(image, filter_type):
    # Convert the image to OpenCV format (numpy array)
    cv_image = np.array(image)

    # Apply the selected filter
    if filter_type == 'grayscale':
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        cv_image = cv2.GaussianBlur(cv_image, (15, 15), 0)
    elif filter_type == 'edge':
        cv_image = cv2.Canny(cv_image, 100, 200)
    elif filter_type == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        cv_image = cv2.transform(cv_image, kernel)
        cv_image = np.clip(cv_image, 0, 255)

    return cv_image 

# Route to handle image upload and filter application
@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    # Get the uploaded file and the selected filter from the form
    image_file = request.files['image']
    filter_type = request.form['filter']

    # Open the image using PIL
    image = Image.open(image_file)

    # Apply the filter to the image
    filtered_image = apply_filter_to_image(image, filter_type)

    # Convert the filtered image to a format that can be sent to the frontend
    if len(filtered_image.shape) == 2:  # For grayscale or edge-detection
        filtered_image = Image.fromarray(filtered_image)
    else:
        filtered_image = Image.fromarray(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))

    img_io = io.BytesIO()
    filtered_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Send the image back to the frontend
    return send_file(img_io, mimetype='image/png')



#task 13 
#fuction to create the s3 bucket 
# Set AWS credentials as environment variables (You can also configure this using AWS CLI or IAM roles)
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''

s3 = boto3.client('s3', region_name='ap-south-1')
# Create S3 bucket
@app.route('/create_bucket', methods=['POST'])
def create_bucket():
    bucket_name = request.form.get('bucket_name')
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            ACL='private',
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-south-1'
            }
        )
        return jsonify({"message": "Bucket created successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/get_buckets', methods=['GET'])
def get_buckets():
    try:
        buckets = s3.list_buckets()
        bucket_names = [bucket['Name'] for bucket in buckets['Buckets']]
        return jsonify({"buckets": bucket_names}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Upload file to S3 bucket
@app.route('/upload_file', methods=['POST'])
def upload_file():
    bucket_name = request.form.get('bucket_name')
    file = request.files['file']

    try:
        s3.upload_fileobj(file, bucket_name, file.filename)
        return jsonify({"message": "File uploaded successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# List files in S3 bucket
@app.route('/list_files', methods=['POST'])
def list_files():
    bucket_name = request.form.get('bucket_name')

    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        files = [content['Key'] for content in response.get('Contents', [])]
        return jsonify({"files": files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# List all buckets
def list_buckets():
    try:
        buckets = s3.list_buckets()
        return [bucket['Name'] for bucket in buckets['Buckets']]
    except Exception as e:
        return []

# Generate presigned URL for download
@app.route('/generate_presigned_url', methods=['POST'])
def generate_presigned_url():
    bucket_name = request.form.get('bucket_name')
    object_name = request.form.get('object_name')

    try:
        url = s3.generate_presigned_url('get_object',
                                        Params={'Bucket': bucket_name, 'Key': object_name},
                                        ExpiresIn=3600)
        return jsonify({"url": url}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#task 14
#fuction for linux terminal
@app.route('/run-terminal-command', methods=['POST'])
def run_terminal_command():
    try:
        data = request.get_json()
        command = data.get('command')

        if not command:
            return jsonify({'error': 'No command provided'}), 400

        # Check the OS and modify the command if necessary
        if os.name == 'nt':  # Windows
            command = command.replace('ls', 'dir')

        # Run the Linux command or Windows command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Return the output (stdout or stderr)
        if result.returncode == 0:
            return jsonify({'output': result.stdout.strip()})
        else:
            return jsonify({'output': result.stderr.strip()}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


#task 15 
# AWS IAM Client setup
iam_client = boto3.client('iam', 
                          aws_access_key_id='add your access key id', 
                          aws_secret_access_key='add your secret key', 
                          region_name='')

@app.route('/create-iam-user', methods=['POST'])
def create_iam_user():
    try:
        # Fetching the user input from the request
        data = request.get_json()
        user_name = data.get('username')
        policy_json = data.get('policy')

        # Create IAM User
        response = iam_client.create_user(UserName=user_name)

        # Attach the policy to the user if a policy is provided
        if policy_json:
            iam_client.put_user_policy(
                UserName=user_name,
                PolicyName=f'{user_name}_policy',
                PolicyDocument=policy_json
            )

        return jsonify({"message": f"IAM User '{user_name}' created successfully", "details": response}), 200
    except ClientError as e:
        return jsonify({"error": str(e)}), 400



#task 16:
#send whatsapp msg
# Twilio credentials (make sure to replace these with your actual credentials)
account_sid = 'your sid'
auth_token = 'your token no.'
whatsapp_from = ''  # Twilio Sandbox WhatsApp number

# Initialize Twilio Client
client = Client(account_sid, auth_token)

@app.route('/send-whatsapp', methods=['POST'])
def send_whatsapp_message():
    try:
        # Fetching the user input from the request
        data = request.get_json()
        to_number = data.get('number')
        message_body = data.get('message')

        # Sending the WhatsApp message via Twilio
        message = client.messages.create(
            body=message_body,
            from_=whatsapp_from,
            to=f'whatsapp:{to_number}'
        )

        return jsonify({"message": "WhatsApp message sent successfully", "sid": message.sid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400



app.run(port="5000",host='0.0.0.0')
