import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import os
from datetime import datetime

def process_image(image_path):
    """
    Process the uploaded image for display and analysis.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Processed image
    """
    # Open image with PIL
    try:
        image = Image.open(image_path)
        
        # Rotate image according to EXIF orientation if present
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            exif = dict(image._getexif().items())
            
            if orientation in exif:
                if exif[orientation] == 2:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif exif[orientation] == 3:
                    image = image.transpose(Image.ROTATE_180)
                elif exif[orientation] == 4:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif exif[orientation] == 5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                elif exif[orientation] == 6:
                    image = image.transpose(Image.ROTATE_270)
                elif exif[orientation] == 7:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                elif exif[orientation] == 8:
                    image = image.transpose(Image.ROTATE_90)
        except (AttributeError, KeyError, IndexError, TypeError):
            # Cases: image doesn't have EXIF data or doesn't have orientation tag
            pass
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if very large (preserve aspect ratio)
        max_dim = 1200
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        return image
    
    except Exception as e:
        print(f"Error processing image: {e}")
        # Return a blank image on error
        return Image.new('RGB', (400, 300), color='gray')

def extract_metadata(image_path):
    """
    Extract useful metadata from the image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary of metadata information
    """
    metadata = {}
    
    try:
        image = Image.open(image_path)
        
        # Basic image info - always include these
        metadata["Image Name"] = os.path.basename(image_path)
        metadata["Width"] = f"{image.width} px"
        metadata["Height"] = f"{image.height} px"
        metadata["Format"] = image.format
        metadata["Size"] = f"{os.path.getsize(image_path) / 1024:.1f} KB"
        metadata["Dimensions"] = f"{image.width} × {image.height} pixels"
        metadata["Color Mode"] = image.mode
        metadata["Aspect Ratio"] = f"{image.width / image.height:.2f}"
        
        # Try to extract EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            exif = {
                ExifTags.TAGS[k]: v
                for k, v in image._getexif().items()
                if k in ExifTags.TAGS
            }
            
            # Extract date/time
            if "DateTime" in exif:
                metadata["Date Taken"] = exif["DateTime"]
            
            # Extract make/model
            if "Make" in exif:
                metadata["Camera Make"] = exif["Make"]
            if "Model" in exif:
                metadata["Camera Model"] = exif["Model"]
                
            # Extract other useful EXIF data
            if "ExposureTime" in exif:
                exposure = exif["ExposureTime"]
                if isinstance(exposure, tuple) and len(exposure) == 2:
                    metadata["Exposure"] = f"{exposure[0]}/{exposure[1]} sec"
                else:
                    metadata["Exposure"] = str(exposure)
                    
            if "FNumber" in exif:
                f_number = exif["FNumber"]
                if isinstance(f_number, tuple) and len(f_number) == 2:
                    metadata["Aperture"] = f"f/{f_number[0]/f_number[1]:.1f}"
                else:
                    metadata["Aperture"] = f"f/{f_number}"
                    
            if "ISOSpeedRatings" in exif:
                metadata["ISO"] = exif["ISOSpeedRatings"]
            
            # Extract GPS info if available
            if "GPSInfo" in exif and exif["GPSInfo"]:
                gps_info = {}
                for key in exif["GPSInfo"].keys():
                    gps_tag = ExifTags.GPSTAGS.get(key, key)
                    gps_info[gps_tag] = exif["GPSInfo"][key]
                
                # Extract latitude
                if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
                    lat = gps_info["GPSLatitude"]
                    lat_ref = gps_info["GPSLatitudeRef"]
                    lat_value = (lat[0] + lat[1]/60 + lat[2]/3600) * (-1 if lat_ref == "S" else 1)
                    metadata["GPS Latitude"] = f"{lat_value:.6f}"
                
                # Extract longitude
                if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
                    lon = gps_info["GPSLongitude"]
                    lon_ref = gps_info["GPSLongitudeRef"]
                    lon_value = (lon[0] + lon[1]/60 + lon[2]/3600) * (-1 if lon_ref == "W" else 1)
                    metadata["GPS Longitude"] = f"{lon_value:.6f}"
        
        return metadata
    
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        # Return basic file info even if EXIF extraction fails
        try:
            image = Image.open(image_path)
            metadata["Image Name"] = os.path.basename(image_path)
            metadata["Width"] = f"{image.width} px"
            metadata["Height"] = f"{image.height} px"
            metadata["Format"] = image.format or "Unknown"
            metadata["Size"] = f"{os.path.getsize(image_path) / 1024:.1f} KB"
            metadata["Dimensions"] = f"{image.width} × {image.height} pixels"
        except:
            metadata["Error"] = "Could not extract basic image information"
        
        return metadata

def enhance_image_quality(image):
    """
    Enhance image quality for better species detection.
    
    Args:
        image (PIL.Image): Original image
        
    Returns:
        PIL.Image: Enhanced image
    """
    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Apply mild denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Apply histogram equalization to improve contrast
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Convert back to PIL format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enhanced_image = Image.fromarray(img)
    
    return enhanced_image

def generate_image_thumbnail(image, size=(300, 300)):
    """
    Generate a thumbnail version of the image.
    
    Args:
        image (PIL.Image): Original image
        size (tuple): Desired thumbnail size
        
    Returns:
        PIL.Image: Thumbnail image
    """
    # Create a copy to avoid modifying the original
    thumb = image.copy()
    
    # Generate thumbnail (preserves aspect ratio)
    thumb.thumbnail(size)
    
    return thumb
