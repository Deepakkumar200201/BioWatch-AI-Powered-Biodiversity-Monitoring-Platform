import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import colorsys
import random

def display_map(locations, zoom_start=5, height=400, include_info=False):
    """
    Display a folium map with the given locations.
    
    Args:
        locations (list): List of dictionaries with latitude and longitude
        zoom_start (int): Initial zoom level
        height (int): Height of the map in pixels
        include_info (bool): Whether to include info popup with location details
        
    Returns:
        None: Displays the map in the Streamlit app
    """
    if not locations:
        st.error("No location data available for map display.")
        return
    
    # Calculate map center
    lat_values = [loc.get("latitude", 0) for loc in locations if loc.get("latitude")]
    lng_values = [loc.get("longitude", 0) for loc in locations if loc.get("longitude")]
    
    if not lat_values or not lng_values:
        st.error("Invalid location data for map display.")
        return
    
    center_lat = sum(lat_values) / len(lat_values)
    center_lng = sum(lng_values) / len(lng_values)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start)
    
    # Add marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Add markers for each location
    for loc in locations:
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        
        if lat and lng:
            # Create marker with popup if info is available
            if include_info and "info" in loc:
                folium.Marker(
                    location=[lat, lng],
                    popup=folium.Popup(loc["info"], max_width=300),
                    icon=folium.Icon(color="green", icon="leaf")
                ).add_to(marker_cluster)
            else:
                folium.Marker(
                    location=[lat, lng],
                    icon=folium.Icon(color="green", icon="leaf")
                ).add_to(marker_cluster)
    
    # Display the map
    folium_static(m, height=height)

def plot_species_distribution(data_df):
    """
    Create a chart showing species distribution.
    
    Args:
        data_df (pd.DataFrame): DataFrame with detection data
        
    Returns:
        alt.Chart: Altair chart object
    """
    # Ensure we have a DataFrame
    if isinstance(data_df, list):
        data_df = pd.DataFrame(data_df)
    
    if len(data_df) == 0:
        # Create empty chart if no data
        return alt.Chart().mark_bar()
    
    # Aggregate data by species
    species_counts = data_df.groupby("species")["count"].sum().reset_index()
    
    # Sort by count descending
    species_counts = species_counts.sort_values("count", ascending=False)
    
    # Create the chart
    chart = alt.Chart(species_counts).mark_bar().encode(
        x=alt.X('species:N', sort='-y', title='Species', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('count:Q', title='Number of Detections'),
        color=alt.Color('species:N', legend=None),
        tooltip=['species', 'count']
    ).properties(
        title='Species Distribution'
    ).interactive()
    
    return chart

def plot_detection_trends(data_df):
    """
    Create a chart showing detection trends over time.
    
    Args:
        data_df (pd.DataFrame): DataFrame with detection data
        
    Returns:
        alt.Chart: Altair chart object
    """
    # Ensure we have a DataFrame
    if isinstance(data_df, list):
        data_df = pd.DataFrame(data_df)
    
    if len(data_df) == 0:
        # Create empty chart if no data
        return alt.Chart().mark_line()
    
    # Convert timestamp to datetime
    data_df["datetime"] = pd.to_datetime(data_df["timestamp"])
    data_df["date"] = data_df["datetime"].dt.date
    
    # Aggregate data by date and species
    daily_counts = data_df.groupby(["date", "species"])["count"].sum().reset_index()
    
    # Create the chart
    chart = alt.Chart(daily_counts).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('count:Q', title='Number of Detections'),
        color=alt.Color('species:N', title='Species'),
        tooltip=['date', 'species', 'count']
    ).properties(
        title='Detection Trends Over Time'
    ).interactive()
    
    return chart

def plot_confidence_distribution(data_df):
    """
    Create a chart showing the distribution of detection confidence scores.
    
    Args:
        data_df (pd.DataFrame): DataFrame with detection data
        
    Returns:
        alt.Chart: Altair chart object
    """
    # Ensure we have a DataFrame
    if isinstance(data_df, list):
        data_df = pd.DataFrame(data_df)
    
    if len(data_df) == 0:
        # Create empty chart if no data
        return alt.Chart().mark_bar()
    
    # Filter out entries with confidence = 0 (no detection)
    data_df = data_df[data_df["confidence"] > 0]
    
    if len(data_df) == 0:
        # Create empty chart if no valid data
        return alt.Chart().mark_bar()
    
    # Create confidence bins
    bin_edges = [0, 0.25, 0.5, 0.75, 1.0]
    bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    # Apply binning
    data_df['confidence_bin'] = pd.cut(
        data_df['confidence'], 
        bins=bin_edges, 
        labels=bin_labels, 
        include_lowest=True
    )
    
    # Count detections by confidence bin and species
    conf_counts = data_df.groupby(['confidence_bin', 'species']).size().reset_index(name='count')
    
    # Create the chart
    chart = alt.Chart(conf_counts).mark_bar().encode(
        x=alt.X('confidence_bin:N', title='Confidence Range'),
        y=alt.Y('count:Q', title='Number of Detections'),
        color=alt.Color('species:N', title='Species'),
        tooltip=['confidence_bin', 'species', 'count']
    ).properties(
        title='Detection Confidence Distribution'
    ).interactive()
    
    return chart

def plot_location_heatmap(data_df):
    """
    Create a heatmap showing species richness by location.
    
    Args:
        data_df (pd.DataFrame): DataFrame with detection data
        
    Returns:
        alt.Chart: Altair chart object
    """
    # Ensure we have a DataFrame
    if isinstance(data_df, list):
        data_df = pd.DataFrame(data_df)
    
    if len(data_df) == 0 or 'location_name' not in data_df.columns:
        # Create empty chart if no data or missing location column
        return alt.Chart().mark_rect()
    
    # Count unique species by location
    location_species = data_df.groupby('location_name')['species'].nunique().reset_index()
    location_species.columns = ['location_name', 'species_count']
    
    # Create the chart
    chart = alt.Chart(location_species).mark_bar().encode(
        x=alt.X('location_name:N', title='Location', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('species_count:Q', title='Number of Unique Species'),
        color=alt.Color('species_count:Q', scale=alt.Scale(scheme='viridis'), title='Species Count'),
        tooltip=['location_name', 'species_count']
    ).properties(
        title='Species Richness by Location'
    ).interactive()
    
    return chart
