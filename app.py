import os
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

import tkinter as tk
from tkinter import filedialog, Button

from smart_image_labeling import sam_utils
from smart_image_labeling import utils

img_buff_x = 0


class ImageEditor:
    def __init__(self, root, default_directory=None):

        # Set the overall window size
        root.geometry("1000x600")

        self.root = root
        self.root.title("Image Annotator")

        # Initialize variables
        self.image_files = []
        self.current_image = None
        self.index = 0

        self.default_directory = default_directory or os.getcwd()

        # Create a Canvas to insert the image
        #self.canvas = tk.Canvas(root, cursor="cross")
        #self.canvas.pack(fill="both", expand=True)

        # Create left panel
        self.panel = tk.Frame(root, width=200, bg='#212121')
        self.panel.pack(side="left", fill="y")

        self.left_buffer = tk.Frame(root, width=100, height=600, bg='#7a7a79')
        self.left_buffer.pack(side='left', fill='y')

        # Set a fixed canvas size
        self.canvas_width = 600
        self.canvas_height = 600

        # Create a Canvas to insert the image with fixed dimensions
        self.canvas = tk.Canvas(
            root, bg='white',
            width=self.canvas_width,
            height=self.canvas_height,
            cursor="cross")
        self.canvas.pack(side='left', fill="both", expand=True)

        self.right_buffer = tk.Frame(root, width=100, height=600, bg='#7a7a79')
        self.right_buffer.pack(side='left', fill='y')

        # buttons
        self.load_button = tk.Button(self.panel, text="Load Directory", command=self.load_directory)
        self.load_button.pack(side="top", pady=(100, 0))  # Padding only at the top
        self.reset_button = tk.Button(self.panel, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side="top", pady=(50, 0))
        self.save_button = tk.Button(self.panel, text="Save Polygons", command=self.save_current_polygons)
        self.save_button.pack(side="top", pady=(50, 0))

        # Bind click event to canvas
        self.canvas.bind("<Button-1>", self.add_point)
        # Bind right and left arrow keys to browse images
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Left>", self.previous_image)

        self.sam_predictor = sam_utils.setup_sam()

        # initialize
        self.current_img_path = None
        self.current_polygons = []
        self.points = []
        self.polygon_ids = []
        self.data = {}
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.output_dir = f'{timestamp}'

    ##############################################################

    def reset_points(self):
        for _, p_id in self.points:
            self.canvas.delete(p_id)
        self.points = []
        self.points = []
        for p_id in self.polygon_ids:
            self.canvas.delete(p_id)
        self.polygon_ids = []
        self.current_polygons = []

    def save_current_polygons(self):
        if not self.current_img_path:
            return
        if not self.current_polygons:
            return
        resized_polygons = utils.resize_polygons(
            self.current_polygons, 1/self.current_scale_factor
        )
        features_dict = utils.save_polygons(
            self.current_img_path,
            resized_polygons,
            self.output_dir
        )

    def load_directory(self):
        # Use directory dialog to select an image directory
        #directory = filedialog.askdirectory()
        directory = filedialog.askdirectory(initialdir=self.default_directory)
        if directory:
            # List all image files in the directory
            self.image_files = [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
            ]
            self.index = 0
            self.load_image()

    def load_image(self):
        self.reset_points()
        if 0 <= self.index < len(self.image_files):
            # Open an image file
            image_path = self.image_files[self.index]
            self.current_img_path = image_path
            img = Image.open(image_path)
            img, scale_factor = utils.resize_image(
                img, self.canvas_width, self.canvas_height)
            self.current_scale_factor = scale_factor

            # Convert the Image object to ImageTk.PhotoImage object
            self.photo = ImageTk.PhotoImage(img)

            # Set canvas size to image size
            self.canvas.config(
                width=self.canvas_width,
                height=self.canvas_height
            )
            self.canvas.create_image(
                (self.canvas_width - img.width) // 2,
                (self.canvas_height - img.height) // 2,
                anchor="nw",
                image=self.photo
            )
            img_array = np.array(img, dtype='uint8')
            self.sam_predictor.set_image(img_array)

    def add_point(self, event):
        # Add a point (circle) where the user clicks
        radius = 7  # Size of the circle
        x1, y1 = (event.x - radius), (event.y - radius)
        x2, y2 = (event.x + radius), (event.y + radius)
        point_id = self.canvas.create_oval(x1, y1, x2, y2, fill="red")
        self.points.append(((event.x, event.y), point_id))
        print('(event.x, event.y):', (event.x - img_buff_x, event.y))

        self.generate_polygons()

    def generate_polygons(self):
        polygons = sam_utils.run_predictor(
            predictor=self.sam_predictor,
            points=[p for p, _ in self.points]
        )
        self.current_polygons = polygons
        for polygon in polygons:
            coords = polygon.exterior.coords
            coords = np.array(coords).astype(np.int32)
            coords[:, 0] += img_buff_x
            print('2:', polygon)
            #flat_coords = [coord for pair in coords for coord in pair]
            flat_coords = coords.flatten().tolist()
            polygon_id = self.canvas.create_polygon(
                flat_coords, outline='red', fill='', width=2)
            self.polygon_ids.append(polygon_id)

    def next_image(self, event):
        # save current polygons
        self.save_current_polygons()

        # Go to next image
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.load_image()

    def previous_image(self, event):
        # save current polygons
        self.save_current_polygons()

        # Go to previous image
        if self.index > 0:
            self.index -= 1
            self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()

