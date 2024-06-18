import numpy as np
from PIL import Image
from IPython.display import display

# # Show image in Jupyter Notebook
# img = Image.open("histogram.png")
        
# # Display image
# display(img)

import os



def skibedy():
    # Generate random int between 0 and 2
    ran = np.random.randint(0, 3)
    
    if ran == 2:
        
        img = Image.open("src\__pycache__\histogram.jpeg")
        
        display(img)
        
    

    