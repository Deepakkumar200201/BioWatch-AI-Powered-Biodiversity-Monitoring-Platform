import numpy as np
import cv2
from PIL import Image
import io
import os
import random
from datetime import datetime

# Constants
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score to consider a detection

# Define a set of common wildlife species with detailed information for classification
SPECIES_DATABASE = {
    "white-tailed deer": {
        "scientific_name": "Odocoileus virginianus",
        "weight_range": "90-310 lbs (41-140 kg)",
        "height_range": "3-3.5 ft (0.9-1.1 m)",
        "habitat": "Forests, grasslands, and farmlands",
        "conservation_status": "Least Concern",
        "diet": "Herbivore - leaves, twigs, fruits, nuts",
        "lifespan": "4-5 years in wild",
        "description": "Common deer species with white underside to tail, raised when alarmed"
    },
    "red fox": {
        "scientific_name": "Vulpes vulpes",
        "weight_range": "8-15 lbs (3.5-7 kg)",
        "height_range": "1-1.3 ft (30-40 cm)",
        "habitat": "Forests, grasslands, mountains, deserts",
        "conservation_status": "Least Concern",
        "diet": "Omnivore - small mammals, birds, fruits, insects",
        "lifespan": "2-5 years in wild",
        "description": "Medium-sized fox with reddish fur and bushy tail with white tip"
    },
    "gray wolf": {
        "scientific_name": "Canis lupus",
        "weight_range": "60-145 lbs (27-65 kg)",
        "height_range": "2.3-2.8 ft (70-85 cm)",
        "habitat": "Forests, mountains, tundra",
        "conservation_status": "Least Concern (globally)",
        "diet": "Carnivore - large hoofed mammals, smaller animals",
        "lifespan": "6-8 years in wild",
        "description": "Largest wild canine species, often travel in packs"
    },
    "black bear": {
        "scientific_name": "Ursus americanus",
        "weight_range": "126-550 lbs (57-250 kg)",
        "height_range": "2.3-3 ft (70-90 cm) at shoulder",
        "habitat": "Forests, swamps, mountains",
        "conservation_status": "Least Concern",
        "diet": "Omnivore - berries, nuts, insects, fish, small mammals",
        "lifespan": "18-25 years in wild",
        "description": "Medium-sized bear with black or brown fur, excellent tree climbers"
    },
    "eastern cottontail rabbit": {
        "scientific_name": "Sylvilagus floridanus",
        "weight_range": "2-4 lbs (0.9-1.8 kg)",
        "height_range": "5-7 inches (12-18 cm)",
        "habitat": "Meadows, farmlands, suburban areas",
        "conservation_status": "Least Concern",
        "diet": "Herbivore - grasses, vegetables, fruits",
        "lifespan": "2-3 years in wild",
        "description": "Small rabbit with grayish-brown fur and white tail underside"
    },
    "eastern gray squirrel": {
        "scientific_name": "Sciurus carolinensis",
        "weight_range": "14-21 oz (400-600 g)",
        "height_range": "7-10 inches (18-25 cm)",
        "habitat": "Deciduous forests, urban parks",
        "conservation_status": "Least Concern",
        "diet": "Omnivore - nuts, seeds, fruits, insects",
        "lifespan": "6-12 years in wild",
        "description": "Common tree squirrel with gray fur and bushy tail"
    },
    "raccoon": {
        "scientific_name": "Procyon lotor",
        "weight_range": "10-30 lbs (4.5-13.5 kg)",
        "height_range": "9-12 inches (23-30 cm)",
        "habitat": "Forests, marshes, urban areas",
        "conservation_status": "Least Concern",
        "diet": "Omnivore - fruits, nuts, insects, small animals",
        "lifespan": "2-3 years in wild",
        "description": "Medium-sized mammal with distinctive black mask and ringed tail"
    },
    "coyote": {
        "scientific_name": "Canis latrans",
        "weight_range": "20-50 lbs (9-23 kg)",
        "height_range": "1.5-2 ft (45-60 cm)",
        "habitat": "Grasslands, forests, urban areas",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - small mammals, birds, fruits",
        "lifespan": "10-14 years in wild",
        "description": "Medium-sized canine with grayish-brown fur, adaptable to various environments"
    },
    "mountain lion": {
        "scientific_name": "Puma concolor",
        "weight_range": "75-175 lbs (34-80 kg)",
        "height_range": "2-2.5 ft (60-75 cm)",
        "habitat": "Mountains, forests, deserts",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - deer, livestock, smaller mammals",
        "lifespan": "8-13 years in wild",
        "description": "Large cat with tawny coat, also known as cougar or puma"
    },
    "bobcat": {
        "scientific_name": "Lynx rufus",
        "weight_range": "15-35 lbs (7-16 kg)",
        "height_range": "1.5-2 ft (45-60 cm)",
        "habitat": "Forests, swamps, deserts",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - rabbits, rodents, birds",
        "lifespan": "7-10 years in wild",
        "description": "Medium-sized cat with spotted coat and short bobbed tail"
    },
    "great horned owl": {
        "scientific_name": "Bubo virginianus",
        "weight_range": "2-5.5 lbs (0.9-2.5 kg)",
        "height_range": "18-25 inches (46-63 cm)",
        "habitat": "Forests, deserts, urban areas",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - small mammals, birds",
        "lifespan": "13-15 years in wild",
        "description": "Large owl with prominent ear tufts, powerful predator"
    },
    "bald eagle": {
        "scientific_name": "Haliaeetus leucocephalus",
        "weight_range": "6.5-14 lbs (3-6.3 kg)",
        "height_range": "2.3-3.3 ft (70-100 cm)",
        "habitat": "Near bodies of water",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - fish, small mammals, birds",
        "lifespan": "20-30 years in wild",
        "description": "Large bird of prey with white head and tail, national symbol of USA"
    },
    "red-tailed hawk": {
        "scientific_name": "Buteo jamaicensis",
        "weight_range": "1.5-3.5 lbs (0.7-1.6 kg)",
        "height_range": "18-26 inches (45-65 cm)",
        "habitat": "Open areas, woodlands",
        "conservation_status": "Least Concern",
        "diet": "Carnivore - small mammals, birds, reptiles",
        "lifespan": "10-15 years in wild",
        "description": "Common hawk with brick-red tail, often seen soaring"
    }
}

# Create a list of species names for the detector
SPECIES_CLASSES = list(SPECIES_DATABASE.keys())

def load_species_detection_model():
    """
    Load a pre-trained model for species detection.
    
    For demonstration purposes, we're using a simplified implementation
    that simulates AI detection capabilities.
    
    Returns:
        A simple model wrapper with detection capabilities
    """
    # In a real implementation, this would load a proper object detection model
    # For demonstration, we'll return a simple object that can be used in our detect_species function
    return {"name": "BioWatch Species Detector", "version": "1.0", "species_classes": SPECIES_CLASSES}

def detect_species(model, image, location=None):
    """
    Simulate detection and classification of species in the provided image.
    
    Args:
        model: The detection model (simple dict in our case)
        image: PIL Image object
        location: Optional dictionary with latitude and longitude
    
    Returns:
        tuple: (detection_results, annotated_image)
            - detection_results: List of dictionaries with species info
            - annotated_image: PIL Image with bounding boxes
    """
    # Convert PIL image to numpy array for processing
    img_array = np.array(image)
    
    # Make a copy for annotation
    annotated_img = img_array.copy()
    height, width = img_array.shape[:2]
    
    # Process results
    detection_results = []
    
    # For demonstration, we'll randomly "detect" 1-3 species with reasonable confidence scores
    num_detections = random.randint(1, 3)
    
    # Define colors for different confidence levels (BGR format for OpenCV)
    high_conf_color = (0, 200, 0)      # Green for high confidence (>0.8)
    medium_conf_color = (0, 165, 255)  # Orange for medium confidence (0.65-0.8)
    low_conf_color = (0, 0, 255)       # Red for lower confidence (<0.65)
    
    # Used species to avoid duplicates in the same image
    used_species = set()
    
    for i in range(num_detections):
        # Randomly select a species that hasn't been used yet
        available_species = [s for s in SPECIES_CLASSES if s not in used_species]
        if not available_species:
            break
            
        species_idx = random.randint(0, len(available_species) - 1)
        species_name = available_species[species_idx]
        used_species.add(species_name)
        
        # Generate a realistic confidence score that's above our threshold
        confidence = random.uniform(CONFIDENCE_THRESHOLD, 0.95)
        
        # Create a reasonable bounding box
        box_x = random.randint(10, width // 2)
        box_y = random.randint(10, height // 2)
        box_w = random.randint(width // 8, width // 3)
        box_h = random.randint(height // 8, height // 3)
        
        # Ensure box stays within image boundaries
        box_x = min(box_x, width - box_w - 5)
        box_y = min(box_y, height - box_h - 5)
        
        # Choose color based on confidence level
        if confidence > 0.8:
            box_color = high_conf_color
        elif confidence > 0.65:
            box_color = medium_conf_color
        else:
            box_color = low_conf_color
        
        # Draw bounding box on the annotated image with thicker lines
        cv2.rectangle(
            annotated_img, 
            (box_x, box_y), 
            (box_x + box_w, box_y + box_h), 
            box_color, 
            3
        )
        
        # Add label with species and confidence on background
        label = f"{species_name.title()}: {confidence:.2f}"
        
        # Calculate label size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for the label
        cv2.rectangle(
            annotated_img,
            (box_x, box_y - label_height - 10),
            (box_x + label_width + 10, box_y),
            box_color,
            -1  # Fill the rectangle
        )
        
        # Add text on top of background
        cv2.putText(
            annotated_img, 
            label, 
            (box_x + 5, box_y - 7), 
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
        
        # Get detailed species info from database
        species_info = SPECIES_DATABASE[species_name]
        
        # Add to detection results with enhanced info
        result = {
            "species": species_name,
            "scientific_name": species_info["scientific_name"],
            "confidence": confidence,
            "count": 1,  # In a real system, this would count individuals
            "bounding_box": [box_x, box_y, box_w, box_h],
            "weight_range": species_info["weight_range"],
            "height_range": species_info["height_range"],
            "habitat": species_info["habitat"],
            "conservation_status": species_info["conservation_status"],
            "description": species_info["description"],
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add location if provided
        if location:
            result.update(location)
            
        detection_results.append(result)
    
    # If by chance we didn't generate any detections, create a "no detection" result
    if not detection_results:
        no_detection = {
            "species": "No wildlife detected",
            "confidence": 0.0,
            "count": 0,
            "bounding_box": None,
            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if location:
            no_detection.update(location)
        detection_results.append(no_detection)
    
    # Convert back to PIL image for display
    annotated_pil = Image.fromarray(annotated_img)
    
    return detection_results, annotated_pil