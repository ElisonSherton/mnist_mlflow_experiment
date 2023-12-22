import pkg_resources
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np

def get_requirements():
    # Get the working set of installed distributions
    working_set = pkg_resources.working_set

    # Iterate through the distributions and get their requirements
    requirements = []
    for dist in working_set:
        requirements.extend(dist.requires())
    requirements = [str(x) for x in requirements]
    return requirements

def get_pillow_image(fig):
    # Create a Matplotlib figure
    # Convert the Matplotlib figure to a Pillow Image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Get the RGB image data from the figure canvas
    image_data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Create a Pillow Image from the RGB data
    pillow_image = Image.fromarray(image_data)

    return pillow_image