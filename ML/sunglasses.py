import cv2
from PIL import Image

# Load your image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()

# Overlay sunglasses on the image
def apply_sunglasses_filter():
    # Load the base image
    base_image = Image.open('captured_image.jpg')

    # Load the sunglasses filter image (transparent PNG) and convert to RGBA
    sunglasses = Image.open('sunglasses.png').convert("RGBA")

    # Resize sunglasses to fit on face
    sunglasses = sunglasses.resize((200, 100))

    # Position the sunglasses on the face
    position = (150, 100)  # Adjust this based on the face's location

    # Create a copy of the base image in RGBA mode
    base_image = base_image.convert("RGBA")

    # Paste the sunglasses onto the base image using sunglasses as the mask
    base_image.paste(sunglasses, position, sunglasses)

    # Show or save the final image
    base_image.show()
    base_image.save('output_with_sunglasses.png')

# Execute the functions in the correct order
capture_image()          # Capture image from webcam
apply_sunglasses_filter() # Apply sunglasses filter after capturing the image
