import streamlit as st
import folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from streamlit_folium import folium_static

# Function to get the user's current location
def get_current_location():
    geolocator = Nominatim(user_agent="streamlit_app")
    try:
        location = geolocator.geocode("my location", timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except GeocoderTimedOut:
        return None

# Streamlit app title and instructions
st.title("Current Location Map")

st.write("This app shows your current location on a map.")

# Get the user's current location
current_location = get_current_location()

if current_location:
    st.write(f"Current Latitude: {current_location[0]}, Longitude: {current_location[1]}")
    
    # Create a folium map centered at the user's current location
    m = folium.Map(location=current_location, zoom_start=15)

    # Add a marker at the current location
    folium.Marker(location=current_location, popup="Your Location").add_to(m)

    # Display the map using streamlit-folium
    folium_static(m)
else:
    st.write("Unable to fetch your current location. Please check your internet connection.")

