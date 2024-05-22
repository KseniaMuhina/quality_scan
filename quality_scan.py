import argparse
import os.path
import cv2
import numpy as np
import fnmatch
from face_sdk_3divi import FacerecService, Config
import pandas as pd
from tqdm import tqdm  # for progress bar


def parse_args():
    """
    Function to parse command line arguments.
    Returns an argparse.Namespace object with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Processing Block Example')
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--num_processed', type=int, default=-1)
    parser.add_argument('--modification', default="assessment", type=str)
    parser.add_argument('--sdk_path', default="../../../", type=str)
    return parser.parse_args()


def find_image_files(directory, n):
    """
    Function to find image files in a directory up to a specified limit.
    Args:
        directory (str): The directory path to search for image files.
        n (int): The maximum number of image files to find. If set to -1 (default), all files will be found.
    Returns:
        list: A list of image file paths found in the directory, up to the specified limit.
    """
    extensions = ['*.png', '*.bmp', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.ppm',
                  '*.PNG', '*.BMP', '*.TIF', '*.TIFF', '*.JPG', '*.JPEG', '*.PPM']
    images_files = []
    for root, dirs, files in os.walk(directory):
        for extension in extensions:
            for filename in fnmatch.filter(files, extension):
                images_files.append(os.path.join(root, filename))
    return images_files[:n] if n != -1 else images_files


def quality_estimator(input_image, sdk_path, modification):
    """
    Estimates the quality of faces in an input image using the Face Recognition SDK.
    Args:
        input_image (str): Path to the input image file.
        sdk_path (str): Path to the directory containing the Face Recognition SDK.
        modification (str): Type of modification to apply during quality estimation.

    Returns:
        dict: A dict containing quality metrics.
        Dictionary contains metrics such as total_score, is_sharp, sharpness_score, etc.
    """
    sdk_conf_dir = os.path.join(sdk_path, 'conf', 'facerec')
    sdk_dll_path = os.path.join(sdk_path, 'lib', 'libfacerec.so')
    sdk_onnx_path = os.path.join(sdk_path, 'lib')

    service = FacerecService.create_service(  # create FacerecService
        sdk_dll_path,
        sdk_conf_dir,
        f'{sdk_path}/license')

    quality_config = {  # quality block configuration parameters
        "unit_type": "QUALITY_ASSESSMENT_ESTIMATOR",  # required parameter
        "modification": modification,
        "ONNXRuntime": {
            "library_path": sdk_onnx_path  # optional
        }
    }
    if modification == "assessment":
        quality_config["config_name"] = "quality_assessment.xml"

    quality_block = service.create_processing_block(
        quality_config)  # create quality assessment estimation processing block

    capturer_config = Config("common_capturer_uld_fda.xml")
    capturer = service.create_capturer(capturer_config)

    img: np.ndarray = cv2.imread(input_image, cv2.IMREAD_COLOR)  # read an image from a file
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert an image in RGB for correct results
    _, im_png = cv2.imencode('.png', image)  # image encoding, required for convertation in raw sample
    img_bytes = im_png.tobytes()  # copy an image to a byte string

    samples = capturer.capture(img_bytes)  # capture faces in an image

    image_ctx = {  # put an image in container
        "blob": image.tobytes(),
        "dtype": "uint8_t",
        "format": "NDARRAY",
        "shape": [dim for dim in image.shape]
    }

    io_data = service.create_context({"image": image_ctx})
    io_data["objects"] = []

    for sample in samples:  # iteration over detected faces in ioData container
        ctx = sample.to_context()
        io_data["objects"].push_back(ctx)  # add results to ioData container

    quality_block(io_data)  # call an estimator and pass a container with a cropped image

    for obj in io_data["objects"]:  # iteration over objects in ioData container
        results_dict = {
            "file_name": input_image,
            "total_score": obj["quality"]["total_score"].get_value(),
            "is_sharp": obj["quality"]["is_sharp"].get_value(),
            "sharpness_score": obj["quality"]["sharpness_score"].get_value(),
            "is_evenly_illuminated": obj["quality"]["is_evenly_illuminated"].get_value(),
            "illumination_score": obj["quality"]["illumination_score"].get_value(),
            "no_flare": obj["quality"]["no_flare"].get_value(),
            "is_left_eye_opened": obj["quality"]["is_left_eye_opened"].get_value(),
            "is_right_eye_opened": obj["quality"]["is_right_eye_opened"].get_value(),
            "is_rotation_acceptable": obj["quality"]["is_rotation_acceptable"].get_value(),
            "not_masked": obj["quality"]["not_masked"].get_value(),
            "is_neutral_emotion": obj["quality"]["is_neutral_emotion"].get_value(),
            "is_eyes_distance_acceptable": obj["quality"]["is_eyes_distance_acceptable"].get_value(),
            "eyes_distance": obj["quality"]["eyes_distance"].get_value(),
            "is_margins_acceptable": obj["quality"]["is_margins_acceptable"].get_value(),
            "is_not_noisy": obj["quality"]["is_not_noisy"].get_value(),
            "has_watermark": obj["quality"]["has_watermark"].get_value(),
            "dynamic_range_score": obj["quality"]["dynamic_range_score"].get_value(),
            "is_dynamic_range_acceptable": obj["quality"]["is_dynamic_range_acceptable"].get_value()
        }
    return results_dict


if __name__ == "__main__":
    args = parse_args()
    files = find_image_files(args.images_path, args.num_processed)
    data = []
    for file in tqdm(files, desc="Processing files"):  # iterating through all files
        data.append(quality_estimator(file, args.sdk_path, args.modification))

    df = pd.DataFrame(data)  # creating a table

    df.to_csv('result.csv')
    print('File result.csv is ready')
