import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import geocoder

from datetime import datetime
from geopy.distance import great_circle
from geopy import distance
from streamlit_option_menu import option_menu
import geopandas as gpd
import pydeck as pdk
from shapely import wkt
from geocodio import GeocodioClient
# import geodesic from geopy
from geopy.distance import geodesic
# import Point from shapely
from shapely.geometry import Point
from pydeck.types import String

geocodio_api = st.secrets["GEOCODIO_API_KEY"]
coord1 = geocoder.ip('me').latlng

st.set_page_config(page_title="Nearby Points of Interest",
                   page_icon="🌎", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)
# set the width of the sidebar
st.markdown("""
<style>
.sidebar .sidebar-content {
    width: 400px;
}
</style>
""", unsafe_allow_html=True)


# Load the CSV file containing points of interest
def load_data():
    df = pd.read_csv('data_v1.csv')
    return df


def distance_2points(row):

    # convert the latitude and longitude to floats
    row['latitude'] = float(row['latitude'])
    row['longitude'] = float(row['longitude'])
    # create a tuple from the lat and long
    coord2 = (row['latitude'], row['longitude'])
    # return the distance in miles
    results = distance.distance(coord1, coord2).miles
    # print(f"Distance: {results}")
    # get the total number of points within the radius

    return results


def highlight_accuracy(val):
    # highlight the accuracy column if the value is 0.50
    color = 'green' if val == 0.50 else 'red'
    return f'background-color: {color}'


def main():
    st.title("Clientes de Power Solar cerca de Mi Ubicacion")
    # Sidebar for selecting distance
    st.sidebar.title("Settings")
    st.sidebar.subheader("Distancia")
    # radio button for selecting between miles and kilometers
    st.sidebar.radio("Selecciona la unidad de distancia:",
                     ["Kilometros", "Millas"], index=0)
    # slider for selecting the distance
    distance = st.sidebar.slider("Selecciona la distancia:", 1, 5, 1, 1)
    # Load the CSV data
    data = load_data()
    # ICON_URL = "https://static-00.iconduck.com/assets.00/solar-panel-icon-2048x1666-6migwmc6.png"
    # ICON_URL = "https://vinte.sh/images/powersolar.png"
    # ICON_URL = "https://powersolarpr.com/wp-content/uploads/2023/06/Icon-SolarPanel-1.png"
    ICON_URL = "https://res.cloudinary.com/javier-jaramillo/image/upload/v1698679542/power_ico.png"
    icon_data = {
        "url": ICON_URL,
        # "path": ICON_PATH,
        "width": 250,
        "height": 250,
        "anchorY": 250,
    }
    # create a button to get the user's current location
    st.sidebar.subheader("Ubicacion")
    st.sidebar.button("Obtener mi ubicacion")
    # coord1 = geocoder.ip('me').latlng
    st.sidebar.write("Tu ubicacion actual es:" + str(coord1))

    # user_location = st.text_input("Enter your address:", "San Juan, Puerto Rico")
    current_location = (18.1388685, -66.2659351)

    client = GeocodioClient(geocodio_api)
    address1 = client.reverse(current_location)
    # parse the json response for the geocodio api
    # st.write(address1)

    # create a dataframe from the json response
    closest_address = pd.DataFrame(address1['results'])
    # remove the columns that are not needed address_components, source
    closest_address = closest_address.drop(
        columns=['address_components', 'source']).copy()
    # split the location column into latitude and longitude
    closest_address[['latitude', 'longitude']] = pd.DataFrame(
        closest_address['location'].tolist(), index=closest_address.index)
    # drop the location column
    closest_address = closest_address.drop(columns=['location'])
    # convert accuracy to a float
    closest_address['accuracy'] = closest_address['accuracy'].astype(float)
    # highlight green if accuracy is 0.50 and red if accuracy is less than 0.50

    # Get user's current location
    # current_location = geocoder.ip('me').latlng
    # fake location from puerto rico
    # use geopandas to reverse geocode the user's location
    st.write("Tu ubicacion actual es:" + str(current_location))
    address = gpd.tools.geocode(current_location, provider='geocodio', user_agent='my-application',
                                timeout=5, api_key=geocodio_api)
    # markdown showing the current location of the user

    # Filter points within the selected distance
    filtered_data = data[data.apply(lambda row: great_circle(
        (row['latitude'], row['longitude']), current_location).miles <= distance, axis=1)]

    filtered_data['distance'] = filtered_data.apply(
        lambda row: distance_2points(row), axis=1)

    # filtered_data['costo_ele'] = filtered_data['costo_ele'].astype(float)

    # st.write(filtered_data.head(10))

    filtered_data['icon_data'] = filtered_data.loc[:, ['latitude', 'longitude']].apply(
        lambda row: icon_data, axis=1)

    cols = st.columns(4)

    with cols[0]:
        # st.write("Name")
        st.metric(label="Numero de Clientes", value=filtered_data.shape[0])

    with cols[1]:
        # st.write("Name")
        population = filtered_data['TotalPop'].mean()
        population = round(population)
        municipality = filtered_data['County'].unique()
        text = f"El municipio de {municipality[0].split(' ')[0]} tiene una poblacion de: "
        st.metric(label=text, value=population)

    with cols[2]:
        income = round(filtered_data['Income'].mean())
        # convert the income to a string with commas
        income = f"{income:,}"
        municipality = filtered_data['County'].unique()
        text = f"El ingreso promedio del municipio de {municipality[0].split(' ')[0]} es de: "
        st.metric(label=text, value=f"${income}")

    with cols[3]:
        income = round(filtered_data['IncomePerCap'].mean())
        # convert the income to a string with commas
        income = f"${income:,}"
        text = f"El ingreso per capita del municipio de {municipality[0].split(' ')[0]} es de: "
        st.metric(label=text, value=income)

    cols = st.columns([0.65, 0.35])
    # st.write(closest_address.style.applymap(highlight_accuracy))
    cols[1].subheader("Direcciones Cercanas")
    cols[1].dataframe(closest_address, use_container_width=True)

    with cols[0]:
        st.subheader("Mapa de Clientes de Energia Solar en Puerto Rico")
        # make sure filtered data is not empty
        if filtered_data.shape[0] == 0:
            st.write(
                f"No hay clientes de energia solar dentro de {distance} kilometros de tu ubicacion actual.")
        else:

            # Display the filtered data on a map
            # plot the points on the map
            layer = pdk.Layer(
                'IconLayer',
                data=filtered_data,
                get_position=["longitude", "latitude"],
                get_icon='icon_data',
                get_size=2,
                size_scale=20,
                get_color='[200, 30, 0, 160]',
                pickable=True,
            )

            # add a layer for the current location
            icon_data = {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Location_icon_from_Noun_Project.png/600px-Location_icon_from_Noun_Project.png?20210513221840",
                "width": 150,
                "height": 150,
                "anchorY": 150,
            }
            current_location_data = pd.DataFrame(
                {'latitude': [current_location[0]], 'longitude': [current_location[1]], 'icon_data': [icon_data]})

            # st.write(current_location_data)
            layer1 = pdk.Layer(
                'IconLayer',
                data=current_location_data,
                get_position=["longitude", "latitude"],
                get_icon='icon_data',
                get_size=4,
                size_scale=18,
                get_color='[255, 140, 0, 160]',
                pickable=True,
            )

            # add a text layer with the text mi ubicacion
            points_layer = pdk.Layer(
                'TextLayer',
                data=current_location_data,
                get_position=["longitude", "latitude"],
                get_color='[20, 30, 0, 160]',
                get_text='Mi Ubicacion',
                get_size=40,
                get_angle=0,
                get_text_anchor='middle',
                get_alignment_baseline='center',
            )

            # add a great circle layer for the distance around the user location
            # convert distance from km to meters
            distance_meters = distance * 1000
            circle_layer = pdk.Layer(
                'PolygonLayer',
                data=[current_location],
                get_polygon='coordinates',
                get_fill_color='[255, 255, 255]',
                filled=True,
                stroked=False,
                get_radius=distance_meters,
                get_line_color=[255, 255, 255],
                get_line_width=1,
                pickable=False,
            )

            # change the zoom level based on the distance
            if distance <= 1:
                zoom = 14
            elif distance <= 2:
                zoom = 13.5
            elif distance <= 3:
                zoom = 13
            elif distance <= 4:
                zoom = 12.5
            else:
                zoom = 12

            # Set the initial map view
            view_state = pdk.ViewState(
                latitude=filtered_data['latitude'].mean(),
                longitude=filtered_data['longitude'].mean(),
                zoom=zoom,
            )

            # Create the map
            map_plot = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                layers=[layer, layer1, points_layer, circle_layer],
                initial_view_state=view_state,
                height=600,
            )

            # Display the map
            st.pydeck_chart(map_plot, use_container_width=True)
            # st.components.v1.html(map_plot.to_html(as_string=True), height=600)

#        st.subheader("Mapa de Posibles Clientes de Energia Solar en Puerto Rico")
#
#        data1 = pd.read_parquet('part-00031-f9394a8f-504e-4ee2-bff7-80ca622ce471.c000.snappy.parquet')
#
#        data1['the_geom_4326'] = data1['the_geom_4326'].apply(wkt.loads)
#
#        gdf = gpd.GeoDataFrame(data1, geometry='the_geom_4326')
#
#
#        # Filter the data by the selected distance by 1 kilometer from the current location
#        filtered_data1 = gdf[gdf.apply(lambda row: great_circle(
#            (row['the_geom_4326'].centroid.y, row['the_geom_4326'].centroid.x), current_location).kilometers <= 1, axis=1)]
#
#        st.write(gdf.shape)
#        st.write(filtered_data.shape)
#
#        # Define a PyDeck layer
#        layer = pdk.Layer(
#            'PolygonLayer',
#            data=filtered_data1,
#            get_polygon='the_geom_4326',
#            # nice viridis color from yellow to red
#            get_fill_color='[255, 255, 255]',
#            filled=True,
#            stroked=False,
#            auto_highlight=True,
        # )
#
#        points_layer = pdk.Layer(
#            'TextLayer',
#            data=filtered_data,
#            get_position=["longitude", "latitude"],
#            get_color='[20, 30, 0, 160]',
#            get_text='client_number',
#            get_size=20,
#            get_angle=0,
#            get_text_anchor='middle',
#            get_alignment_baseline='center',
#        #)

#        # Set the initial map view
#        view_state = pdk.ViewState(
#            latitude=filtered_data1['the_geom_4326'].centroid.y.mean(),
#            longitude=filtered_data1['the_geom_4326'].centroid.x.mean(),
#            zoom=11,
#        #)

#        # Create the map
#        map_plot = pdk.Deck(
#            map_style='mapbox://styles/mapbox/light-v9',
#            layers=[layer, points_layer],
#            initial_view_state=view_state,
#        #)

#        # Display the map
#        st.pydeck_chart(map_plot)


def dashboard():
    st.title("Solar Power Dashboard")
    st.subheader("Electricity Rates in Puerto Rico")

    # here PR metrics will be displayed
    cols = st.columns(3)

    # sidebar selection settings
    st.sidebar.title("Settings")
    show_data = st.sidebar.checkbox("Show Data", value=False)
    show_map = st.sidebar.checkbox("Show Map", value=True)
    show_graph = st.sidebar.checkbox("Show Graph")

    with cols[0]:
        st.metric(label="Poblacion Total", value="3,263,584", delta="0.1%")
    with cols[1]:
        st.metric(label="Total Consumption", value="9,351,471 MWh",
                  delta="2.87 MWh per capita")
    with cols[2]:
        st.metric(label="CO2 Emissions from Consumption",
                  value="6,774,852,385 kg", delta="2,075.89 kg per capita")

    cols = st.columns(3)
    with cols[0]:
        st.metric(label="Total Production", value="17,098,878 MWh",
                  delta="5.24 MWh per capita")
    with cols[1]:
        st.metric(label="Total Production from Renewable",
                  value="325,488 MWh", delta="0.1 MWh per capita")
    with cols[2]:
        st.metric(label="Total Production from Non-Renewable",
                  value="16,774,772 MWh", delta="5.14 MWh per capita")

    st.subheader("Active Clients of Solar Power")
    # data for the graph
    data = load_data()
    cols = st.columns(3)

    with cols[0]:
        # count the number of clients from the data
        st.metric(label="Total Clients", value=data.shape[0])
    with cols[1]:
      # average population per client
        total_population = format(round(data['TotalPop'].mean()), ",")
        st.metric(label="Poblacion Promedio de Puerto Rico",
                  value=total_population)
    with cols[2]:
        income = format(round(data['Income'].mean()), ",")
        st.metric(label="Ingreso Promedio de Puerto Rico", value=f"${income}")

    if show_data:
        st.subheader("Data")
        col1, col2 = st.columns([0.60, 0.40])

        # show a video in the first column
        col1.subheader("Como funciona la energia solar")
        col1.video("https://youtu.be/wmWYpGrJfjg")

        col2.markdown("""
                      # Soluciones para
                      ## APARTAMENTOS
                      Con nuestras baterías portátiles EcoFlow, puedes mantener tu apartamento conectado durante un apagón, sin ruidos ni emisiones.
                      
                      # Soluciones para

                      ## SU NEGOCIO
                      Haz que tu negocio brille con nuestras soluciones de energía renovable, ahorrando en costos de energía y garantizado un servicio ininterrumpido.
                      """)
        col2.image(
            "https://powersolarpr.com/wp-content/uploads/2018/06/PS-Logo.jpg")

    if show_map:

        # st.dataframe(data.head(10))
        # Load in the JSON data
        DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"
        json = pd.read_json(DATA_URL)
        df = pd.DataFrame()

        # st.json(json['features'][0])

        # Custom color scale
        COLOR_RANGE = [
            [65, 182, 196],
            [127, 205, 187],
            [199, 233, 180],
            [237, 248, 177],
            [255, 255, 204],
            [255, 237, 160],
            [254, 217, 118],
            [254, 178, 76],
            [253, 141, 60],
            [252, 78, 42],
            [227, 26, 28],
            [189, 0, 38],
            [128, 0, 38],
        ]

        BREAKS = [-0.6, -0.45, -0.3, -0.15, 0, 0.15,
                  0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]

        def color_scale(val):
            for i, b in enumerate(BREAKS):
                if val < b:
                    return COLOR_RANGE[i]
            return COLOR_RANGE[i]

        def calculate_elevation(val):
            return math.sqrt(val) * 10

        # Parse the geometry out in Pandas
        df["coordinates"] = json["features"].apply(
            lambda row: row["geometry"]["coordinates"])
        df["valuePerSqm"] = json["features"].apply(
            lambda row: row["properties"]["valuePerSqm"])
        df["growth"] = json["features"].apply(
            lambda row: row["properties"]["growth"])
        df["elevation"] = json["features"].apply(
            lambda row: calculate_elevation(row["properties"]["valuePerSqm"]))
        df["fill_color"] = json["features"].apply(
            lambda row: color_scale(row["properties"]["growth"]))

        # Add sunlight shadow to the polygons
        sunlight = {
            "@@type": "_SunLight",
            "timestamp": 1564696800000,  # Date.UTC(2019, 7, 1, 22),
            "color": [255, 255, 255],
            "intensity": 1.0,
            "_shadow": True,
        }

        ambient_light = {"@@type": "AmbientLight",
                         "color": [255, 255, 255], "intensity": 1.0}

        lighting_effect = {
            "@@type": "LightingEffect",
            "shadowColor": [0, 0, 0, 0.5],
            "ambientLight": ambient_light,
            "directionalLights": [sunlight],
        }

        data['exits_radius'] = data.apply(lambda row: great_circle(
            (row['latitude'], row['longitude']), (18.1388685, -66.2659351)).miles, axis=1)

        data = data[data['exits_radius'] <= 100]
        # sort by exits radius
        data = data.sort_values(by=['exits_radius'], ascending=False)

        data['elevation'] = data['exits_radius'].apply(lambda x: x * 2)
        data['fill_color'] = data['elevation'].apply(
            lambda x: [255, 140, 0, 160] if x > 1000 else [255, 140, 0, 160])
        data['line_color'] = data['elevation'].apply(
            lambda x: [255, 140, 0, 160] if x > 1000 else [255, 140, 0, 160])
        data['line_width'] = data['elevation'].apply(
            lambda x: 1 if x > 1000 else 1)
        data['elevation_scale'] = data['elevation'].apply(
            lambda x: 100 if x > 1000 else 100)
        data['radius'] = data['elevation'].apply(
            lambda x: 50 if x > 1000 else 50)
        data['size_scale'] = data['elevation'].apply(
            lambda x: 10 if x > 1000 else 10)
        data['radius_scale'] = data['elevation'].apply(
            lambda x: 30 if x > 1000 else 30)
        data['radius_min_pixels'] = data['elevation'].apply(
            lambda x: 1 if x > 1000 else 1)

        # create an arrays of arrays of coordinates from the latitude and longitude columns in which each array is a polygon
        data['coordinates'] = data.apply(
            lambda row: [[[row['longitude'], row['latitude']]]], axis=1)

        st.subheader("Mapa de Clientes de Energia Solar en Puerto Rico")

        # st.dataframe(data.head(10))

        layer = pdk.Layer(
            'HeatmapLayer',
            data=data,
            opacity=0.9,
            get_position=["longitude", "latitude"],
            aggregation=pdk.types.String("MEAN"),
            # color_range=COLOR_BREWER_BLUE_SCALE,
            threshold=0.1,
            get_weight='id',
            pickable=True,
        )

        points_layer = pdk.Layer(
            "PolygonLayer",
            data=data,
            id="geojson",
            opacity=0.8,
            stroked=False,
            get_polygon="coordinates",
            filled=True,
            extruded=True,
            wireframe=True,
            get_elevation="elevation",
            get_fill_color="fill_color",
            get_line_color=[255, 255, 255],
            auto_highlight=True,
            pickable=True,
        )

        scatterplot = pdk.Layer(
            'ScatterplotLayer',
            data=data,
            pickable=True,
            opacity=0.9,
            stroked=True,
            filled=True,
            radius_scale=30,
            radius_min_pixels=1,
            radius_max_pixels=100,
            line_width_min_pixels=1,
            get_position=["longitude", "latitude"],
            get_radius="exits_radius",
            get_fill_color=[255, 140, 0, 160],
            get_line_color=[0, 0, 0],

        )

        # Set the initial map view
        view_state = pdk.ViewState(
            latitude=data['latitude'].mean(),
            longitude=data['longitude'].mean(),
            zoom=8,
        )

        # Create the map
        map_plot = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[scatterplot],
            initial_view_state=view_state,
        )

        # Display the map
        st.pydeck_chart(map_plot, use_container_width=True,
                        )

    if show_graph:
        st.subheader(
            "Grafica de Clientes de Energia Solar en Puerto Rico por Municipio")
        import plotly.express as px
        import matplotlib.pyplot as plt
        import squarify
        import numpy as np

        municipios_count = [m for m in data['County'].value_counts()]
        municipios = [m for m in data['County'].value_counts().index]

        volume = municipios_count
        labels = municipios
        # color scale for the treemap chart (green to red) create a color list
        color_list = [color for color in plt.cm.Reds(
            np.arange(0, 1, 1/len(labels)))]

        # create a plotly colorscale with the color list
        fig = plt.figure(figsize=(12, 8))

        fig = px.treemap(data, path=['Municipio'], values='TotalPop', color='TotalPop',
                         color_continuous_scale="algae", title='Clientes de Energia Solar en Puerto Rico por Municipio')

        st.plotly_chart(fig, theme="streamlit", use_container_width=True, config={
                        'displayModeBar': False})


def neighborhood():
    st.title("Clientes de Energia Solar en Puerto Rico por Municipio")

    data = load_data()
    # set county to string
    data['County'] = data['County'].astype(str)
    # create a new column with the county name without the word Municipio
    data['County1'] = data['County'].apply(
        lambda x: x.replace(" Municipio", ""))
    # create a list of unique municipios
    unique_municipios = data['County1'].unique().tolist()
    # add the option for all municipios
    # unique_municipios.insert(0, ("Todos", "Todos"))
    # st.write(unique_municipios)

    # Sidebar for selecting distance

    st.sidebar.title("Settings")
    municipio = st.sidebar.selectbox(
        "Selecciona el Municipio:", sorted(unique_municipios), index=0)

    # st.write(municipio)
    filtered_data = data[data['County1'] == municipio]

    # merger filtered data with with electricity consumption data
    consumption_data = pd.read_csv('consumption.csv')

    filtered_data = filtered_data.merge(
        consumption_data, how='left', left_on='County', right_on='County')

    # st.write(filtered_data.shape)

    start_lat = filtered_data['latitude'].mean()
    start_lon = filtered_data['longitude'].mean()

    usa = gpd.read_file('cb_2013_us_county_500k.geojson')
    usa = gpd.read_file(
        'cb_2018_us_county_500k/cb_2018_us_county_500k.shp')
    # usa = gpd.read_file('cb_2013_us_county_500k.geojson')

    usa = usa[usa['LSAD'] == '13']
    json_municipio = usa[usa['NAME'] == municipio]
    # st.write(json_municipio)
    # convert geopandas dataframe to geoseires
    # json_municipio = json_municipio.__geo_interface__.copy()
    # print(json_municipio)
    # st.write(json_municipio)

    centroids = gpd.GeoDataFrame(
        json_municipio, geometry='geometry')
    centroids.set_geometry('geometry')
    centroids = centroids.to_crs("EPSG:4326")
    centroids["geometry"] = json_municipio.geometry.centroid
    centroids["Name"] = json_municipio.NAME
    # centroids["geometry"] = json_municipio['features'][0]['geometry']['coordinates']
    # centroids["Name"] = json_municipio['features'][0]['properties']['NAME']
    # calculate the centroid of the polygon
    # centroids["centroid"] = centroids["geometry"].apply(
    #    lambda x: Point(np.mean(x[0], axis=0), np.mean(x[1], axis=0)))
    # st.write(centroids)

    # json_municipio = json_municipio.to_crs("EPSG:4326")
    # json_municipio['centroid'] = json_municipio['geometry'].centroid
    # st.write(json_municipio)

    # centroids["geometry"] = json_municipio.geometry.centroid
    # centroids["Name"] = json_municipio.Name

    # st.map(centroids, zoom=8)

    # st.dataframe(filtered_data)

    st.subheader(
        f"Estadisticas de Consumo de Energia en el Municipio de {municipio}")

    cols = st.columns(4)
    total_consumption = format(round(filtered_data['comp0'].mean()), ",")
    cols[0].metric(label="Total Consumption",
                   value=total_consumption, delta="MWh")
    consumption_per_capita = format(
        round(filtered_data['comp1'].mean(), 2), ",")
    cols[1].metric(label="Consumption per Capita", value=consumption_per_capita,
                   delta="MWh")
    co2_emissions = format(round(filtered_data['comp2'].mean()), ",")
    cols[2].metric(label="CO2 Emissions", value=co2_emissions, delta="kg")
    c02_emissions_per_capita = format(
        round(filtered_data['comp3'].mean()), ",")
    cols[3].metric(label="CO2 Emissions per Capita", value=c02_emissions_per_capita,
                   delta="kg")

    st.subheader(f"Clientes de Energia Solar en el Municipio de {municipio}")
    cols = st.columns(3)
    total_clients = filtered_data.shape[0]
    # st.write(filtered_data.shape)
    cols[0].metric(label="Total Clients", value=total_clients)
    total_population = filtered_data['TotalPop'].mean()
    cols[1].metric(label=f"El Municipio de {municipio} tiene una poblacion de:",
                   value=f"{round(total_population):,}")
    total_income = filtered_data['Income'].mean()
    cols[2].metric(label=f"El Municipio de {municipio} tiene un Ingreso Promedio de:",
                   value=f"${round(total_income):,}")

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        tooltip={"text": "{Name}"},
        initial_view_state=pdk.ViewState(
            latitude=start_lat,
            longitude=start_lon,
            zoom=11,
            pitch=30,
        ),

        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                data=json_municipio,
                opacity=0.8,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation="TotalPop",
                get_fill_color="[255, 255, 255]",
                get_line_color=[255, 255, 255],
                pickable=True,
            ),
            pdk.Layer(
                "ColumnLayer",
                data=filtered_data,
                get_position=["longitude", "latitude"],
                get_color='[255, 140, 0, 160]',
                get_elevation='costo_ele',
                elevation_scale=100,
                get_fill_color='[255, 140, 0, 160]',
                get_line_color='[255, 140, 0, 160]',
                # elevation_range=[0, 10000000],
                size_scale=10,
                radius=50,
                pickable=True,
                extruded=True,
                cell_size=200,
                # auto_highlight=True,
            ),
            pdk.Layer(
                "GridLayer",
                data=filtered_data,
                get_position=["longitude", "latitude"],
                get_color='[255, 140, 0, 160]',
                elevation_scale=50,
                elevation_range=[0, 100],
                size_scale=int(3),
                radius=int(15),
                # pickable=True,
                # extruded=bool(True),
                auto_highlight=True,

            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_data,
                get_position=["longitude", "latitude"],
                get_color='[255, 140, 0, 160]',
                get_radius=0,
                opacity=0.5,
            ),
            pdk.Layer(
                'GeoJsonLayer',
                data=json_municipio,
                get_fill_color=[0, 0, 0, 100],
            ),
            pdk.Layer(
                'TextLayer',
                data=centroids,
                get_position="geometry.coordinates",
                get_size=18,
                get_color=[255, 255, 255],
                get_text="Name",
                get_angle=0,
                get_text_anchor=String("middle"),
                get_alignment_baseline=String("center"),
            ),
        ],
    ))


def otras_herramientas():
    # add Puerto Rico Solar Map: Generation and Storage Projects
    st.subheader("Puerto Rico Solar Map: Generation and Storage Projects")
    st.markdown("""
                This map shows solar and storage projects installed at critical facilities since Hurricane Maria in September 2017. The majority were funded by humanitarian and philanthropic organizations. The names of these Funders/Implementers and their proportional contribution can be found in the pie charts to the left. They can also be accessed through the drop-down menu on the top right. Totals for Puerto Rico are given in the widgets at the top of the map.
                To view data by Funder/Implementer, select the organization of interest on the pie chart (please note the top pie chart has a second tab called "Other" Funders/Implementers where additional organizations can be found). Data for your selected organization will appear in the right-hand widgets. These tabs will remain blank unless a Funder/Implementer is selected.
                To de-select click outside the pie chart or scroll to the top of the drop down list and select "All".
                You can view individual project data by selecting dots on the map
                """)
    data1 = pd.read_csv('PR_Solar_Data_for_Mapping_Open_Data.csv')

    # calculate the exit radius base on status Complete, In Progress, and Pending and size of the project
    data1['exits_radius'] = data1.apply(lambda row: 100 if row['Status'] == 'Complete' else (
        50 if row['Status'] == 'In Process' else 25), axis=1)

    cols = st.columns([0.25, 0.50, 0.25])
    with cols[0]:
        total_projects = data1.shape[0]
        st.metric(label="Total Projects", value=total_projects)
        total_pv = data1['PV'].sum()
        st.metric(label="Total PV", value=total_pv)

    with cols[1]:

        # st.dataframe(data1)
        # here map will be displayed
        layer = pdk.Layer(
            "ScatterplotLayer",
            data1,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=6,
            radius_min_pixels=1,
            radius_max_pixels=100,
            line_width_min_pixels=1,
            get_position=["Longitude", "Latitude"],
            get_radius="exits_radius",
            get_fill_color=[255, 140, 0],
            get_line_color=[0, 0, 0],
        )

        # Set the initial map view
        view_state = pdk.ViewState(
            latitude=data1['Latitude'].mean(),
            longitude=data1['Longitude'].mean(),
            zoom=8,
        )

        # Create the map
        map_plot = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            layers=[layer],
            initial_view_state=view_state,
        )

        # Display the map
        st.pydeck_chart(map_plot, use_container_width=True)

    with cols[2]:
        # here graph will be displayed
        pass


if __name__ == "__main__":
    menu_option = ["Inicio", "Municipios",
                   "Mi Ubicacion", "Otras Herramientas"]
    selected2 = option_menu(None, menu_option,
                            icons=['house', 'cloud-upload',
                                   "list-task", 'gear'],
                            menu_icon="cast", default_index=2, orientation="horizontal")

    # selected2 = "Mi Ubicacion"

    st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(205, 104, 0);
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: black;
}



</style>
""", unsafe_allow_html=True)

    if selected2 == "Inicio":
        dashboard()
    elif selected2 == "Municipios":
        neighborhood()
    elif selected2 == "Mi Ubicacion":
        main()
    elif selected2 == "Otras Herramientas":
        otras_herramientas()
