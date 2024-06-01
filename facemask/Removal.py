from contextlib import nullcontext
import cv2
import os
import imutils
import logging
import sys
import ctypes
import numpy as np
from io import BytesIO
from PIL import Image, ImageTk
from keras.models import load_model
import skimage
import tensorflow as tf
import torchvision.transforms as T
import torch

from django.core.files.uploadedfile import InMemoryUploadedFile
import tkinter as tk
import tk_tools
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter import *
from PIL import Image, ImageTk
import tempfile

# Adjusted parameters for eye detection and mask creation
EYE_SCALE_FACTOR = 1.2
EYE_MIN_NEIGHBORS = 5
MASK_THRESHOLD = 32
MIN_LINE_LENGTH = 1
MAX_LINE_GAP = 2



def facial_Feature(image, gray):
    eye_cascade = cv2.CascadeClassifier('C:/Users/USER/OneDrive/Desktop/Django/facemask/haarcascade_eye.xml')
    
    roi_gray = gray
    roi_color = image
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=EYE_SCALE_FACTOR, minNeighbors=EYE_MIN_NEIGHBORS)
    eye_y = []

    for (ex, ey, ew, eh) in eyes:
        eye_y.append(ey + (eh / 2))

    return roi_color, roi_gray, eye_y

def line_Getter(img, gray, eye_avg, threshold, minLine, maxGap):
    blurred_gray = cv2.GaussianBlur(gray, (7,7),0)
    edges = cv2.Canny(blurred_gray, 26, 135)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=minLine, maxLineGap=maxGap)

    w = img.shape[1]
    final_lines = []
    if lines is not None: 
        for line in lines: 
            x1, y1, x2, y2 = line[0]
            if y1 >= eye_avg and y2 >= eye_avg and 0 + 50 < x1 < w - 50 and 0 + 50 < x2 < w - 50:
                final_lines.append(line[0])
    return final_lines


def mask_Creator(lines, img):
    y_max = -1
    y_min = 99999
    x_max = -1
    x_min = 9999999

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 != 0:
            slope = abs(y2 - y1) / abs(x2 - x1)
        else:
            slope = 2
        if slope < 1 and max(y1, y2) >= y_max:
            y_max = max(y1, y2)
        
        if min(y2, y1) <= y_min:
            y_min = min(y2, y1)

    for line in lines:
        x1, y1, x2, y2 = line
        y_max_temp = y_max
        if x2 - x1 != 0:
            slope = abs(y2 - y1) / abs(x2 - x1)
        else:
            slope = 0
        if max(x1, x2) >= x_max and (y2 < y_max_temp and y1 < y_max_temp) and slope > 1:
            x_max = max(x1, x2)
        if min(x1, x2) <= x_min and (y2 < y_max_temp and y1 < y_max_temp) and slope > 1:
            x_min = min(x1, x2)

    mask_img = np.zeros(img.shape, dtype="uint8")
    h,w=mask_img.shape[:2]
    cv2.rectangle(mask_img, (x_min, y_max), (x_max, y_min - 19), (255, 255, 255), -1)
    return mask_img



def load_images_from_folder(path):
    
    images = []
    
    for item in os.listdir(path):
        img = cv2.imread(os.path.join(path,item))
       
        if img is not None:
            images.append(img)

    return images



def output_Creator(cropped_image, mask_img):
    # # set up network
   
    generator_state_dict = torch.load("C:/Users/USER/OneDrive/Desktop/Django/facemask/pretrained/states_pt_celebahq.pth", map_location= torch.device('cpu'))['G']


    from facemask.model.networks import Generator
 

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')
    # # # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    mask_pil = Image.fromarray(mask)
    # load image and mask
    image = im_pil
    mask = mask_pil

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask
    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    #img_out.show()
    
    return img_out

   

def reject_outliers(data, m=6.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()




# MAIN FUNCTION starting function

globalimage=['']
class MaskRemoval(object):
    def resize_image(self, image_file, target_size=(256, 256)):
        print('entering the resize function')
        if isinstance(image_file, InMemoryUploadedFile):
            # Convert the image file to a numpy array
            nparr = np.frombuffer(image_file.read(), np.uint8)
            # Decode the image array to an OpenCV image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Resize the image to the target size
            resized_image = cv2.resize(image, target_size)
            print('Image resized')
            return resized_image
        else:
            return None


    def upload_file(self, image_file):
        global globalimage
        print('Image has been sent')
        print('img in upload file: ', image_file)
        img = self.resize_image(image_file)
        
        globalimage[0] = img
        imagelist = self.output_file()  # Get the list of predicted images
        # temp_files = []  # List to hold temporary file paths
        # for idx, image_data in enumerate(imagelist):
        #     # Create a temporary file object
        #     temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        #     # Write the image data to the temporary file
        #     temp_file.write(image_data)
        #     # Close the file to ensure data is flushed to disk
        #     temp_file.close()
        #     # Add the path of the temporary file to the list
        #     temp_files.append(temp_file.name)
        return imagelist
    # def upload_file(self,image_file):
    #     global globalimage
            
    #     print('Image has been sent')
    #     img = self.resize_image(image_file)
    #     globalimage.append(img)
    #     output = self.output_file()  # Pass the resized image to output_filey
    #     if isinstance(output, np.ndarray):
    #         # Encode the output image as JPEG
    #         ret, jpeg = cv2.imencode('.jpg', output)
    #         if ret:
    #             # Return the bytes representation of the JPEG image
    #             return jpeg.tobytes()
        
    #     # Return None if the output image data is not valid
    #     return None



    # def upload_file():
    #     global tkImage
    #     global b2
    #     global globalimage
    #     global frame

    #     filename = filedialog.askopenfilename(filetypes=[("Image file", ".jpg .png")])

    #     image=Image.open(filename)
    #     img=image.resize((256, 256))
    #     tkImage = ImageTk.PhotoImage(img)
        
    #     img = cv2.imread(filename)
    #     globalimage.append(img)
    #     frame = Frame(window, width=900, height=600)
    #     frame.grid(column=0, row=7)
    #     # frame.place(anchor='center', relx=0  , rely=)
    #     b2 = tk.Button(window, image=tkImage, anchor="center")
    #     #b2.grid(row=3,column=0)
    #     b2.grid(column=0, row=5)


    def output_file(self):
        images = globalimage
        print('image has received to output_file function')
        labels_dict = {0: 'without mask', 1: 'mask'}
        color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
        size = 4
        # classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = images[0]
        # print(image.shape)
        # cv2.imshow("Input",imagePath)
        inter = cv2.INTER_AREA
        height = 256
        dim = None
        (h, w) = image.shape[:2]
        r = height / float(h)
        dim = (int(w * r), height)
        image = cv2.resize(image, dim, interpolation=inter)

        # image = cv2.resize(image, (512,512), interpolation = inter)

        # cv2.imwrite("OUTPUT/image.png",image)
        # image = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size)) # make a smaller image
        print(image.shape)
        # detect MultiScale / faces
        # faces = classifier.detectMultiScale(mini)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Input", image)
        cropped_image, cropped_gray, eye_y =facial_Feature(image, gray)
        if len(eye_y) > 0:

            # Remove outliner eyes
            eye_y = np.array(eye_y)
            eye_y = reject_outliers(eye_y)
            print(f"EYE Y: {eye_y}")
            eye_avg = (sum(eye_y) / len(eye_y)) + 24  # get average of eyes and look just below

        else:
            print("NO EYES")
            eye_avg = int(cropped_image.shape[0] / 2) + 12
        imagelist = []

        for x in range(2,5):
            y = x + 3
            minlength = 9 - y
            if minlength >= 0:
                minlength = 1
            # final_lines =line_Getter(cropped_image, cropped_gray, eye_avg, 32 - y, minlength, 2 + y)
            # mask_img =mask_Creator(final_lines, cropped_image)

            # image2 = (output_Creator(cropped_image, mask_img))

            # imagelist.append(image2)

            # z = 1
            # if x % 2 == 0:
            #     z = -1
            final_lines = line_Getter(cropped_image, cropped_gray, eye_avg - x, 32 - y, minlength, 2 + y)
            mask_img = mask_Creator(final_lines, cropped_image)
            image2 = (output_Creator(cropped_image, mask_img))

            imagelist.append(image2)

        return imagelist


    def reset(globalimage):
        globalimage.clear()
