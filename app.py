import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
import uuid

# Import modules from our application
from models import load_species_detection_model, detect_species
from utils import get_sample_locations, format_confidence_score
from image_processor import process_image, extract_metadata
from data_manager import save_detection_results, load_detection_history
from visualization import display_map, plot_species_distribution, plot_detection_trends

# Page configuration
st.set_page_config(
    page_title="BioWatch: AI Biodiversity Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .dashboard-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 28px;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = None
if 'current_location' not in st.session_state:
    st.session_state.current_location = None
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Load the detection model
@st.cache_resource
def get_model():
    return load_species_detection_model()

model = get_model()

# Sidebar
with st.sidebar:
    st.title("BioWatch 🌿")
    st.write("AI-powered biodiversity monitoring platform")
    
    # Theme toggle
    st.subheader("Settings")
    theme_options = {"light": "Light Mode 🌞", "dark": "Dark Mode 🌙"}
    selected_theme = st.selectbox(
        "Application Theme",
        options=list(theme_options.keys()),
        format_func=lambda x: theme_options[x],
        index=0 if st.session_state.theme == "light" else 1
    )
    
    # Apply theme change if needed
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        # Apply custom CSS for dark mode if needed
        if selected_theme == "dark":
            st.markdown("""
            <style>
                body {
                    background-color: #121212;
                    color: #ffffff;
                }
                .metric-container {
                    background-color: #1e1e1e !important;
                }
                .stTabs [data-baseweb="tab"] {
                    background-color: #2a2a2a;
                }
                .css-1544g2n {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                    background-color: #121212;
                }
                .element-container, .stTextInput > div, label, .stSelectbox > div {
                    color: #ffffff !important;
                }
            </style>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.subheader("Navigation")
    page = st.radio("Navigation Pages", ["Upload & Detect", "Data Dashboard", "Reports", "About"], label_visibility="collapsed")
    
    # Location selector
    st.subheader("Camera Trap Location")
    locations = get_sample_locations()
    selected_location = st.selectbox(
        "Select monitoring location",
        options=list(locations.keys())
    )
    st.session_state.current_location = locations[selected_location]
    
    # Display a small map of the selected location
    st.write("Selected Location:")
    display_map(
        [st.session_state.current_location], 
        zoom_start=10,
        height=200
    )
    
    st.markdown("---")
    st.info("BioWatch uses AI to identify species in camera trap images and monitor biodiversity.")

# Main content area based on selected page
if page == "Upload & Detect":
    st.header("Species Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload section
        st.subheader("Upload Camera Trap Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process the image and store in session state
            image = process_image(tmp_path)
            st.session_state.uploaded_image = image
            st.session_state.processed_image = image  # Will be updated with bounding boxes
            
            # Display the original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Extract metadata
            metadata = extract_metadata(tmp_path)
            if metadata:
                st.write("Image Metadata:")
                for key, value in metadata.items():
                    st.write(f"- **{key}**: {value}")
            
            # Remove the temp file
            os.unlink(tmp_path)
    
    with col2:
        st.subheader("Detection Results")
        
        if st.session_state.uploaded_image is not None:
            # Button to perform detection
            if st.button("Detect Species"):
                with st.spinner("Analyzing image..."):
                    # Get detection results
                    results, annotated_image = detect_species(
                        model, 
                        st.session_state.uploaded_image,
                        location=st.session_state.current_location
                    )
                    
                    # Update session state
                    st.session_state.detection_results = results
                    st.session_state.processed_image = annotated_image
                    
                    # Save results to history
                    detection_id = str(uuid.uuid4())
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_detection_results(
                        detection_id,
                        timestamp,
                        uploaded_file.name,
                        st.session_state.current_location,
                        results
                    )
                    
                    # Reload detection history
                    st.session_state.detection_history = load_detection_history()
                    
                    st.success("Detection completed!")
            
            # Display results if available
            if st.session_state.detection_results:
                # Display annotated image
                st.image(st.session_state.processed_image, caption="Detection Results", use_column_width=True)
                
                # Display detected species with enhanced information
                st.write("## Detected Species")
                
                # Create a dataframe for the basic results
                detection_df = pd.DataFrame(st.session_state.detection_results)
                
                # Format confidence for display
                detection_df["confidence"] = detection_df["confidence"].apply(format_confidence_score)
                
                # For each detection, show detailed information in expandable sections
                for i, detection in enumerate(st.session_state.detection_results):
                    species_name = detection["species"].title()
                    if species_name == "No Wildlife Detected":
                        st.info("No wildlife species were detected in this image.")
                        continue
                        
                    confidence = format_confidence_score(detection["confidence"])
                    confidence_color = "green" if detection["confidence"] > 0.8 else "orange" if detection["confidence"] > 0.65 else "red"
                    
                    # Create an expander for each detected species
                    with st.expander(f"{species_name} ({confidence})", expanded=i==0):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Scientific Name:** *{detection['scientific_name']}*")
                            st.markdown(f"**Weight Range:** {detection['weight_range']}")
                            st.markdown(f"**Height/Length:** {detection['height_range']}")
                            st.markdown(f"**Conservation Status:** {detection['conservation_status']}")
                        
                        with col2:
                            st.markdown(f"**Habitat:** {detection['habitat']}")
                            st.markdown(f"**Confidence:** <span style='color:{confidence_color}'>{confidence}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Detected At:** {detection['detected_at']}")
                        
                        st.markdown("---")
                        st.markdown(f"**Description:** {detection['description']}")
                        
                # Create a tabular view with basic information
                st.write("### Summary Table")
                
                # Rename columns for display
                summary_df = detection_df.rename(columns={
                    "species": "Species",
                    "scientific_name": "Scientific Name",
                    "confidence": "Confidence",
                    "count": "Count"
                })
                
                # Display as table 
                if "No wildlife detected" not in summary_df["Species"].values:
                    st.table(summary_df[["Species", "Scientific Name", "Confidence", "Count"]])
                
                # Add download button for results with more comprehensive data
                csv = detection_df.to_csv(index=False)
                st.download_button(
                    label="Download detailed results as CSV",
                    data=csv,
                    file_name="detection_results.csv",
                    mime="text/csv",
                )
        else:
            st.info("Please upload an image to begin species detection.")

elif page == "Data Dashboard":
    st.header("Biodiversity Monitoring Dashboard")
    
    # Load detection history if not already loaded
    if not st.session_state.detection_history:
        st.session_state.detection_history = load_detection_history()
    
    # If there's no detection history, show a message
    if not st.session_state.detection_history or len(st.session_state.detection_history) == 0:
        st.info("No detection data available. Upload and analyze images to populate the dashboard.")
    else:
        # Dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        # Convert detection history to DataFrame for analysis
        history_df = pd.DataFrame(st.session_state.detection_history)
        
        with col1:
            total_species = history_df["species"].nunique()
            st.metric("Total Species Detected", total_species)
        
        with col2:
            total_detections = history_df["count"].sum()
            st.metric("Total Animal Detections", int(total_detections))
        
        with col3:
            total_images = history_df["image_name"].nunique()
            st.metric("Analyzed Images", total_images)
        
        # Species distribution
        st.subheader("Species Distribution")
        species_chart = plot_species_distribution(history_df)
        st.altair_chart(species_chart, use_container_width=True)
        
        # Temporal trends
        st.subheader("Detection Trends Over Time")
        trends_chart = plot_detection_trends(history_df)
        st.altair_chart(trends_chart, use_container_width=True)
        
        # Map of all detection locations
        st.subheader("Monitoring Locations")
        all_locations = history_df[["latitude", "longitude"]].drop_duplicates().to_dict("records")
        display_map(all_locations)
        
        # Raw data
        st.subheader("Detection History")
        st.dataframe(
            history_df[["timestamp", "image_name", "species", "confidence", "count"]],
            use_container_width=True
        )

elif page == "Reports":
    st.header("Biodiversity Monitoring Reports")
    
    # Load detection history if not already loaded
    if not st.session_state.detection_history:
        st.session_state.detection_history = load_detection_history()
    
    if not st.session_state.detection_history or len(st.session_state.detection_history) == 0:
        st.info("No data available for reports. Upload and analyze images first.")
    else:
        # Convert detection history to DataFrame for analysis
        history_df = pd.DataFrame(st.session_state.detection_history)
        
        # Report options
        report_type = st.selectbox(
            "Select Report Type",
            options=["Species Summary", "Location Analysis", "Detection Timeline"]
        )
        
        if report_type == "Species Summary":
            st.subheader("Species Summary Report")
            
            # Group by species and calculate metrics
            species_summary = history_df.groupby("species").agg(
                total_count=("count", "sum"),
                avg_confidence=("confidence", "mean"),
                detections=("detection_id", "count")
            ).reset_index()
            
            # Format for display
            species_summary["avg_confidence"] = species_summary["avg_confidence"].apply(
                lambda x: f"{x:.1%}"
            )
            species_summary["total_count"] = species_summary["total_count"].astype(int)
            
            # Rename columns
            species_summary.columns = ["Species", "Total Count", "Avg. Confidence", "Detection Events"]
            
            # Display summary
            st.table(species_summary)
            
            # Species distribution chart
            st.subheader("Species Distribution")
            species_chart = plot_species_distribution(history_df)
            st.altair_chart(species_chart, use_container_width=True)
            
            # Export options
            csv = species_summary.to_csv(index=False)
            st.download_button(
                "Download Species Summary",
                data=csv,
                file_name="species_summary_report.csv",
                mime="text/csv"
            )
            
        elif report_type == "Location Analysis":
            st.subheader("Location Analysis Report")
            
            # Group by location
            location_summary = history_df.groupby(["latitude", "longitude"]).agg(
                unique_species=("species", "nunique"),
                total_detections=("count", "sum"),
                images_analyzed=("image_name", "nunique")
            ).reset_index()
            
            # Display the summary
            st.write("Monitoring Location Summary:")
            location_df = location_summary.copy()
            location_df.columns = ["Latitude", "Longitude", "Unique Species", "Total Detections", "Images Analyzed"]
            st.table(location_df)
            
            # Map visualization of locations with species counts
            st.subheader("Biodiversity Map")
            st.write("Map showing monitoring locations colored by species richness:")
            
            # Create location markers with species count info
            locations = []
            for _, row in location_summary.iterrows():
                locations.append({
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "info": f"Species: {row['unique_species']}, Detections: {row['total_detections']}"
                })
            
            display_map(locations, include_info=True)
            
            # Export options
            csv = location_df.to_csv(index=False)
            st.download_button(
                "Download Location Summary",
                data=csv,
                file_name="location_analysis_report.csv",
                mime="text/csv"
            )
            
        elif report_type == "Detection Timeline":
            st.subheader("Detection Timeline Report")
            
            # Convert timestamp to datetime
            history_df["datetime"] = pd.to_datetime(history_df["timestamp"])
            
            # Group by date
            history_df["date"] = history_df["datetime"].dt.date
            timeline_data = history_df.groupby("date").agg(
                species_count=("species", "nunique"),
                total_detections=("count", "sum"),
                images=("image_name", "nunique")
            ).reset_index()
            
            # Display timeline chart
            st.line_chart(
                timeline_data.set_index("date")[["species_count", "total_detections"]]
            )
            
            # Display tabular data
            st.write("Daily Detection Summary:")
            timeline_display = timeline_data.copy()
            timeline_display.columns = ["Date", "Species Count", "Total Detections", "Images Analyzed"]
            st.table(timeline_display)
            
            # Export options
            csv = timeline_display.to_csv(index=False)
            st.download_button(
                "Download Timeline Data",
                data=csv,
                file_name="detection_timeline_report.csv",
                mime="text/csv"
            )

elif page == "About":
    st.header("About BioWatch")
    
    st.write("""
    ## AI-Powered Biodiversity Monitoring
    
    BioWatch is a platform designed to help conservationists, researchers, and citizen scientists 
    monitor and track wildlife using camera trap images. By leveraging artificial intelligence and 
    computer vision technologies, BioWatch can automatically identify species in camera trap images, 
    helping to streamline the biodiversity monitoring process.
    
    ### Key Features:
    
    - **Automated Species Detection**: Upload camera trap images to automatically identify wildlife species
    - **Biodiversity Monitoring**: Track species presence and distribution across different locations
    - **Interactive Dashboard**: Visualize detection data with charts and maps
    - **Report Generation**: Create and export reports for conservation research and planning
    
    ### How It Works:
    
    1. Upload camera trap images from your monitoring locations
    2. BioWatch's AI model analyzes the images to detect and identify species
    3. View detection results with bounding boxes highlighting detected animals
    4. Explore the data dashboard to analyze species distribution and trends
    5. Generate reports for research, conservation planning, and grant applications
    
    ### Technologies Used:
    
    - TensorFlow/Keras for deep learning species detection
    - OpenCV for image processing
    - Streamlit for the interactive web application
    - Pandas for data analysis
    - Plotly and Altair for interactive visualizations
    - Folium for map visualizations
    """)
    
    st.subheader("Get Started")
    st.write("""
    To begin using BioWatch, navigate to the "Upload & Detect" page and upload a camera trap image.
    The system will analyze the image and provide detection results.
    
    For any questions or feedback, please contact the development team.
    """)

# Footer
st.markdown("---")
st.caption("BioWatch: AI-Powered Biodiversity Monitoring | © 2025")
