import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

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

    # Convert the buffer to a PIL image
    img = Image.open(buf)
    return img

# Example usage
if __name__ == "__main__":
    house_image = generate_house_image()
    house_image.show()  # This will display the generated house image
