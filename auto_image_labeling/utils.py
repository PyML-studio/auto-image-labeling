import os
import json
import logging

from PIL import Image
import cv2
from shapely.geometry import Polygon
from shapely import affinity

logger = logging.getLogger('utils')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def resize_image(image, new_width, new_height):
    orig_width, orig_height = image.size

    # scaling width
    w1 = new_width
    r1 = w1 / orig_width
    h1 = int(r1 * orig_height)
    # sacling height
    h2 = new_height
    r2 = h2 / orig_height
    w2 = int(r2 * orig_width)
    if h1 <= new_height and w2 <= new_width:
        if h1 * w1 > h2 * w2:
            scale_factor = r1
        else:
            scale_factor = r2
    elif h1 <= new_height:
        scale_factor = r1
    else:
        scale_factor = r2
    
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
    logger.info(f'resize_polygons: scale_factor={scale_factor}')
    resized_polygons = []
    for polygon in polygons:
        resized_polygon = affinity.scale(
            polygon, xfact=scale_factor, yfact=scale_factor,
            origin=(0, 0)
        )
        # round coordinates
        rounded_coords = [
            (int(round(x)), int(round(y)))
            for x, y in resized_polygon.exterior.coords
        ]
        resized_polygons.append(Polygon(rounded_coords))
        logger.info(f'resized_polygon: {resized_polygon}')

    return resized_polygons


def save_polygons(image_path, polygons, output_dir):
    unique_id = os.path.basename(image_path)[:-4]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    polygons_past = []
    output_filename = os.path.join(output_dir, f'{unique_id}.json')
    if os.path.exists(output_filename):
        with open(output_filename, 'rt') as fp_in:
            d_past = json.load(fp_in)
            polygons_past.extend(d_past['polygons'])
    else:
        polygons_past = []

    list_wkt = [polygon.wkt for polygon in polygons]
    data = {
        'image_path': image_path,
        'polygons': polygons_past + list_wkt
    }

    with open(output_filename, 'wt') as fp:
        json.dump(data, fp, indent=4)
