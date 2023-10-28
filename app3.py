import streamlit as st
import pandas as pd
import time
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

geocodio_api = st.secrets["GEOCODIO_API_KEY"]

st.set_page_config(page_title="Nearby Points of Interest",
                   page_icon=":earth_americas:", layout="wide", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 5px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: red;
}
</style>
""", unsafe_allow_html=True)

# Load the CSV file containing points of interest


def load_data():
    df = pd.read_csv('data_v1.csv')
    return df


def distance_2points(row):
    coord1 = geocoder.ip('me').latlng

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
    ICON_URL = "https://static-00.iconduck.com/assets.00/solar-panel-icon-2048x1666-6migwmc6.png"
    icon_data = {
        "url": ICON_URL,
        "width": 50,
        "height": 50,
        "anchorY": 50,
    }

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
        columns=['address_components', 'source'])
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

    filtered_data['icon_data'] = None
    for i in filtered_data.index:
        filtered_data['icon_data'][i] = icon_data

    # st.dataframe(filtered_data['distance'])

    # Add a column with the distance from the user's location
    cols = st.columns(4)

    with cols[0]:
        # st.write("Name")
        st.metric(label="Numero de Clientes", value=filtered_data.shape[0])

    with cols[1]:
        # st.write("Name")
        population = filtered_data['TotalPop'].mean()
        population = round(population)
        municipality = filtered_data['County'].unique()
        text = f"El municipio de {municipality[0]} tiene una poblacion de:"
        st.metric(label=text, value=population)

    with cols[2]:
        income = round(filtered_data['Income'].mean())
        # convert the income to a string with commas
        income = f"{income:,}"
        municipality = filtered_data['County'].unique()
        text = f"{municipality[0]} tiene un ingreso promedio de: "
        st.metric(label=text, value=f"${income}")

    with cols[3]:
        income = round(filtered_data['IncomePerCap'].mean())
        # convert the income to a string with commas
        income = f"${income:,}"
        text = f"El ingreso per capita promedio es de: "
        st.metric(label=text, value=income)

    cols = st.columns([0.65, 0.35])
    # st.write(closest_address.style.applymap(highlight_accuracy))
    cols[1].subheader("Direcciones Cercanas")
    cols[1].write(closest_address, use_container_width=True)

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
                size_scale=15,
                get_color='[200, 30, 0, 160]',
                pickable=True,
            )

            # add a layer for the current location
            icon_data = {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Location_icon_from_Noun_Project.png/600px-Location_icon_from_Noun_Project.png?20210513221840",
                "width": 50,
                "height": 50,
                "anchorY": 50,
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
                get_color='[200, 30, 0, 160]',
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

            # Set the initial map view
            view_state = pdk.ViewState(
                latitude=filtered_data['latitude'].mean(),
                longitude=filtered_data['longitude'].mean(),
                zoom=14,
            )

            # Create the map
            map_plot = pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                layers=[layer, layer1, points_layer, circle_layer],
                initial_view_state=view_state,
            )

            # Display the map
            st.pydeck_chart(map_plot, use_container_width=True)

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
    show_data = st.sidebar.checkbox("Show Data")
    show_map = st.sidebar.checkbox("Show Map")
    show_graph = st.sidebar.checkbox("Show Graph")

    with cols[0]:
        st.metric(label="Population", value="3,263,584", delta="0.1%")
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
        st.metric(label="Average Population per Client",
                  value=round(data['TotalPop'].mean()))
    with cols[2]:
        st.metric(label="Average Consumption per Client",
                  value=round(data['Income'].mean()))

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
        st.subheader("Mapa de Clientes de Energia Solar en Puerto Rico")
        st.map(data, zoom=8)

    if show_graph:
        st.subheader(
            "Grafica de Clientes de Energia Solar en Puerto Rico por Municipio")
        import plotly.express as px
        import matplotlib.pyplot as plt
        import squarify
        import numpy as np

        municipios_count = [m for m in data['Municipio'].value_counts()]
        municipios = [m for m in data['Municipio'].value_counts().index]

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

    st.subheader("Mapa de Clientes de Energia Solar en Puerto Rico")

    data = load_data()

    # print(data.columns)

    unique_municipios = data['Municipio'].unique()

    st.sidebar.title("Settings")
    municipio = st.sidebar.selectbox(
        "Select Municipio:", sorted(unique_municipios))

    filtered_data = data[data['Municipio'] == municipio]

    print(filtered_data.shape)

    start_lat = filtered_data['latitude'].mean()
    start_lon = filtered_data['longitude'].mean()

    print(start_lat, start_lon)

    usa = gpd.read_file('cb_2013_us_county_500k.geojson')
    usa = usa[usa['LSAD'] == '13']
    json_municipio = usa[usa['Name'] == municipio]
    json_municipio = json_municipio.to_crs("EPSG:4326")
    centroids = gpd.GeoDataFrame()
    centroids["geometry"] = json_municipio.geometry.centroid
    centroids["Name"] = json_municipio.Name

    # st.map(centroids, zoom=8)

    cols = st.columns(3)

    total_clients = filtered_data.shape[0]
    cols[0].metric(label="Total Clients", value=total_clients)
    total_population = filtered_data['TotalPop'].mean()
    cols[1].metric(label="Average Population per Client",
                   value=round(total_population))
    total_consumption = filtered_data['Income'].mean()
    cols[2].metric(label="Average Consumption per Client",
                   value=round(total_consumption))

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
                get_color='[200, 30, 0, 160]',
                get_elevation='costo_ele',
                elevation_scale=100,
                get_fill_color='[100, 30, 0, 160]',
                get_line_color='[100, 30, 0, 160]',
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
                get_color='[20, 30, 0, 160]',
                elevation_scale=50,
                elevation_range=[0, 100],
                size_scale=int(5),
                radius=int(20),
                # pickable=True,
                # extruded=bool(True),
                auto_highlight=True,

            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_data,
                get_position=["longitude", "latitude"],
                get_color='[200, 30, 0, 160]',
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
                get_size=14,
                get_color=[255, 255, 255],
                get_text="Name",
                get_angle=0,
            ),
        ],
    ))


if __name__ == "__main__":
    menu_option = ["Inicio", "Municipios",
                   "Mi Ubicacion", "Otras Herramientas"]
    selected2 = option_menu(None, menu_option,
                            icons=['house', 'cloud-upload',
                                   "list-task", 'gear'],
                            menu_icon="cast", default_index=0, orientation="horizontal")

    selected2 = "Mi Ubicacion"

    if selected2 == "Inicio":
        dashboard()
    elif selected2 == "Municipios":
        neighborhood()
    elif selected2 == "Mi Ubicacion":
        main()
