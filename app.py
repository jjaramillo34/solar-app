from flask import Flask, request, jsonify
from geopy.distance import geodesic
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Sample user's current location (latitude and longitude)
user_location = (40.7128, -74.0060)  # New York City as an example

@app.route('/geofence', methods=['GET'])
def geofence():
    # Get the radius in miles from the query parameter
    radius = float(request.args.get('radius', 1))

    # Define a list of geofence points (latitude, longitude)
    geofence_points = [(40.7128, -74.0060)]  # Example: New York City

    # Calculate distances to the geofence points
    within_geofence = []
    for point in geofence_points:
        distance = geodesic(user_location, point).miles
        if distance <= radius:
            within_geofence.append(point)

    return jsonify({'within_geofence': within_geofence})

if __name__ == '__main__':
    app.run(debug=True)
