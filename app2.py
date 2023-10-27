import streamlit as st
#from geopy.geocoders import Nominatim
from geopy.geocoders import Photon
import folium
from folium import plugins
from geopy.distance import geodesic
from streamlit_folium import folium_static

# Function to get nearby points based on distance
def get_nearby_points(current_location, points, max_distance):
    nearby_points = []
    for point in points:
        distance = geodesic(current_location, point).miles
        if distance <= max_distance:
            nearby_points.append(point)
    return nearby_points

# Streamlit UI
st.title("Nearby Points of Interest")

# Get user's current location
geolocator = Photon(user_agent="Nearby Points of Interest") # Replace with your own user agent
location = geolocator.geocode("368 9th Avenue, Manhathan NY, 11420")  # Replace with user's current location

if location is not None:
    st.write("Your current location:")
    st.write(f"Latitude: {location.latitude}, Longitude: {location.longitude}")

    # Create a map centered at the user's location
    m = folium.Map(location=[location.latitude, location.longitude], zoom_start=15)

    # Add a marker for the user's location
    folium.Marker([location.latitude, location.longitude], tooltip="Your Location").add_to(m)

    # Define some example points of interest (replace with your data)
    points_of_interest = [
        (location.latitude + 0.01, location.longitude + 0.01),
        (location.latitude + 0.005, location.longitude + 0.005),
        (location.latitude - 0.01, location.longitude - 0.01),
    ]

    # Select the radius for nearby points (1 mile, 2 miles, or 5 miles)
    max_distance = st.selectbox("Select Radius (in miles)", [1, 2, 5])

    # Get nearby points
    nearby_points = get_nearby_points((location.latitude, location.longitude), points_of_interest, max_distance)

    # Add nearby points to the map
    for point in nearby_points:
        folium.Marker(point, icon=folium.Icon(color='green')).add_to(m)

    # Display the map
    folium_static(m)
else:
    st.write("Unable to determine your current location.")
