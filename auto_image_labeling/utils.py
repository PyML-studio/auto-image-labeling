import os
from PIL import Image
import cv2
from shapely.geometry import Polygon
from shapely import affinity
import json


def resize_image(image, new_width, new_height):
    orig_width, orig_height = image.size
    
    orig_aspect = orig_width / orig_height
    new_aspect = new_width / new_height
    
    # Scale the image while preserving the aspect ratio
    if orig_aspect >= new_aspect:
        # Original image is wider
        scale_factor = new_width / orig_width
    else:
        # Original image is taller
        scale_factor = new_height / orig_height
    
    # Calculate new dimensions
    resized_width = int(orig_width * scale_factor)
    resized_height = int(orig_height * scale_factor)
    
    # Resize and return the image
    resized_image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS)
    return resized_image, scale_factor


def polygonize(
        binary_array,
        tolerance=2.0,
        area_threshold=20
    ):
    contours, _ = cv2.findContours(
        binary_array,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) < 4:
            continue
        polygon = Polygon(contour.squeeze()) #.reshape(-1, 2))

        if not polygon.is_empty and polygon.area > area_threshold:
            #simplified_coords = np.array(polygon.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
            polygons.append(polygon.simplify(tolerance))

    return polygons


def resize_polygons(polygons, scale_factor):
    print('resize_polygons:', scale_factor)
    resized_polygons = []
    for polygon in polygons:
        resized_polygon = affinity.scale(
            polygon, xfact=scale_factor, yfact=scale_factor,
            origin=(0, 0)
        )
        resized_polygons.append(resized_polygon)
        print('resized_polygon:', resized_polygon)
    return resized_polygons


def save_polygons(image_path, polygons, output_dir):
    unique_id = os.path.basename(image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    polygons_past = []
    output_filename = os.path.join(output_dir, f'{unique_id}.json')
    if os.path.exists(output_filename):
        with open(output_filename, 'rt') as fp_in:
            d_past = json.load(fp_in)
            polygons_past.extend(d_past['geometries'])
    else:
        polygons_past = []

    list_wkt = [polygon.wkt for polygon in polygons + polygons_past]
    data = {
        'image_path': image_path,
        'geometries': list_wkt
    }

    with open(output_filename, 'wt') as fp:
        json.dump(data, fp, indent=4)
