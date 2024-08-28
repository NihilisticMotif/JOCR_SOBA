########################################################################################################################################################

# Before Refactoring (Repeated Code):

def process_image_type_a(img_path, width, height):
    # Common setup code
    img = Path2Image(img_path)
    processed_img = apply_filter_a(img, width, height)
    text = extract_text(processed_img)
    return text

def process_image_type_b(img_path, width, height):
    # Common setup code with slight variation
    img = Path2Image(img_path)
    processed_img = apply_filter_b(img, width, height)
    text = extract_text(processed_img)
    return text

def process_image_type_c(img_path, width, height):
    # Another slight variation
    img = Path2Image(img_path)
    processed_img = apply_filter_c(img, width, height)
    text = extract_text(processed_img)
    return text

########################################################################################################################################################

# After Refactoring (Generalized Function):

from typing import Callable

def process_image(
    img_path: str, 
    width: int, 
    height: int, 
    filter_function: Callable[[np.ndarray, int, int], np.ndarray]
) -> str:
    
    '''
    Generalized function to process an image with a specific filter.

    Parameters:
        img_path (str): Path to the image file.
        width (int): Width parameter for processing.
        height (int): Height parameter for processing.
        filter_function (Callable): The filtering function to apply on the image.

    Returns:
        str: Extracted text from the processed image.
    '''
    
    img = Path2Image(img_path)
    processed_img = filter_function(img, width, height)
    text = extract_text(processed_img)
    return text

# Now you can call this generalized function with different filters
text_a = process_image('path/to/image_a.png', 100, 200, apply_filter_a)
text_b = process_image('path/to/image_b.png', 100, 200, apply_filter_b)
text_c = process_image('path/to/image_c.png', 100, 200, apply_filter_c)


########################################################################################################################################################
