import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import random

def get_sample_locations():
    """
    Returns a dictionary of sample monitoring locations for the application.
    
    Returns:
        dict: Dictionary mapping location names to coordinate dictionaries
    """
    return {
        "Yellowstone North": {
            "latitude": 44.9631, 
            "longitude": -110.5989,
            "location_name": "Yellowstone North"
        },
        "Yellowstone Central": {
            "latitude": 44.4280, 
            "longitude": -110.5885,
            "location_name": "Yellowstone Central"
        },
        "Yellowstone South": {
            "latitude": 44.1350, 
            "longitude": -110.6663,
            "location_name": "Yellowstone South"
        },
        "Grand Teton": {
            "latitude": 43.7904, 
            "longitude": -110.6818,
            "location_name": "Grand Teton"
        },
        "Olympic National Forest": {
            "latitude": 47.8021, 
            "longitude": -123.6044,
            "location_name": "Olympic National Forest"
        },
        "Yosemite Valley": {
            "latitude": 37.7456, 
            "longitude": -119.5936,
            "location_name": "Yosemite Valley"
        },
        "Glacier National Park": {
            "latitude": 48.7596, 
            "longitude": -113.7870,
            "location_name": "Glacier National Park"
        },
        "Everglades": {
            "latitude": 25.2866, 
            "longitude": -80.8987,
            "location_name": "Everglades"
        },
        "Great Smoky Mountains": {
            "latitude": 35.6131, 
            "longitude": -83.5532,
            "location_name": "Great Smoky Mountains"
        }
    }

def get_data_directory():
    """
    Returns the directory where application data should be stored.
    Creates the directory if it doesn't exist.
    
    Returns:
        str: Path to the data directory
    """
    # Use the current directory for simplicity
    data_dir = os.path.join(os.getcwd(), "data")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    return data_dir

def format_confidence_score(confidence):
    """
    Format confidence score as a percentage with 1 decimal place.
    
    Args:
        confidence (float): Confidence score between 0 and 1
        
    Returns:
        str: Formatted percentage string
    """
    return f"{confidence:.1%}"

def timestamp_to_datetime(timestamp_str):
    """
    Convert timestamp string to datetime object.
    
    Args:
        timestamp_str (str): Timestamp string in format "%Y-%m-%d %H:%M:%S"
        
    Returns:
        datetime: Datetime object
    """
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def get_date_range_from_history(history):
    """
    Get start and end dates from detection history.
    
    Args:
        history (list): List of detection history dictionaries
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if not history:
        # Return default range if no history
        end_date = datetime.now()
        start_date = end_date.replace(day=1)
        return start_date, end_date
    
    # Extract timestamps and convert to datetime
    timestamps = [timestamp_to_datetime(record["timestamp"]) for record in history]
    
    return min(timestamps), max(timestamps)

def validate_image_format(file):
    """
    Validate that the file is an image with supported format.
    
    Args:
        file: File object from Streamlit file_uploader
        
    Returns:
        bool: True if valid, False otherwise
    """
    if file is None:
        return False
    
    # Check file extension
    valid_extensions = ['.jpg', '.jpeg', '.png']
    file_ext = os.path.splitext(file.name.lower())[1]
    
    return file_ext in valid_extensions
