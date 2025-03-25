import os
import json
import pandas as pd
from datetime import datetime
import uuid
from utils import get_data_directory

# Default history file path
HISTORY_FILE = os.path.join(get_data_directory(), "detection_history.json")

def save_detection_results(detection_id, timestamp, image_name, location, results):
    """
    Save detection results to the history file.
    
    Args:
        detection_id (str): Unique ID for this detection
        timestamp (str): Timestamp when detection was performed
        image_name (str): Name of the uploaded image
        location (dict): Location information (latitude, longitude)
        results (list): List of detection result dictionaries
    """
    # Load existing history
    history = load_detection_history()
    
    # Add new results to history
    for result in results:
        # Create entry for each detected species with enhanced data
        entry = {
            "detection_id": detection_id,
            "timestamp": timestamp,
            "image_name": image_name,
            "species": result["species"],
            "confidence": result["confidence"],
            "count": result.get("count", 1),  # Default to 1 if count not provided
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "location_name": location.get("location_name", "Unknown")
        }
        
        # Add detailed species information if available
        if "scientific_name" in result:
            entry["scientific_name"] = result["scientific_name"]
        if "weight_range" in result:
            entry["weight_range"] = result["weight_range"]
        if "height_range" in result:
            entry["height_range"] = result["height_range"]
        if "conservation_status" in result:
            entry["conservation_status"] = result["conservation_status"]
        if "habitat" in result:
            entry["habitat"] = result["habitat"]
        if "description" in result:
            entry["description"] = result["description"]
        if "detected_at" in result:
            entry["detected_at"] = result["detected_at"]
        
        history.append(entry)
    
    # Save updated history
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_detection_history():
    """
    Load detection history from file.
    
    Returns:
        list: List of detection result dictionaries
    """
    if not os.path.exists(HISTORY_FILE):
        # Create empty history file if it doesn't exist
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return []
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        # Return empty list if file is corrupt or unreadable
        return []

def get_species_summary():
    """
    Generate summary statistics for detected species.
    
    Returns:
        pd.DataFrame: DataFrame with species summary statistics
    """
    history = load_detection_history()
    
    if not history:
        # Return empty DataFrame if no history
        return pd.DataFrame(columns=["species", "count", "avg_confidence", "detections"])
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)
    
    # Group by species and calculate metrics
    summary = df.groupby("species").agg(
        count=("count", "sum"),
        avg_confidence=("confidence", "mean"),
        detections=("detection_id", "count")
    ).reset_index()
    
    return summary

def get_location_summary():
    """
    Generate summary statistics for monitoring locations.
    
    Returns:
        pd.DataFrame: DataFrame with location summary statistics
    """
    history = load_detection_history()
    
    if not history:
        # Return empty DataFrame if no history
        return pd.DataFrame(columns=["location_name", "latitude", "longitude", "species_count", "total_detections"])
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(history)
    
    # Group by location and calculate metrics
    summary = df.groupby(["location_name", "latitude", "longitude"]).agg(
        species_count=("species", "nunique"),
        total_detections=("count", "sum")
    ).reset_index()
    
    return summary

def export_detection_data(start_date=None, end_date=None, format="csv"):
    """
    Export detection data for a given date range.
    
    Args:
        start_date (datetime, optional): Start date for filtering
        end_date (datetime, optional): End date for filtering
        format (str): Export format ('csv' or 'json')
        
    Returns:
        str: Path to the exported file
    """
    history = load_detection_history()
    
    if not history:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Convert timestamp strings to datetime for filtering
    df["datetime"] = pd.to_datetime(df["timestamp"])
    
    # Apply date filters if provided
    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date)]
    
    # Remove datetime column (was just for filtering)
    df = df.drop(columns=["datetime"])
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_data_directory()
    
    if format.lower() == "csv":
        filename = f"detection_export_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
    else:  # json
        filename = f"detection_export_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        df.to_json(output_path, orient="records", indent=2)
    
    return output_path

def clear_detection_history():
    """
    Clear all detection history.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Write empty list to history file
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        return True
    except:
        return False
