import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set page config first
st.set_page_config(
    page_title="AI-ML Decision System for Effective Farming",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Home page
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Define sidebar options
sidebar_options = ["Home Page", "Crop Recommendation", "Fertilizer Recommendation", "Crop Yield Prediction"]
page_selection = st.sidebar.radio("Go to", sidebar_options)

# Update session state based on sidebar selection
if page_selection == "Crop Recommendation":
    st.session_state['page'] = 'crop_recommendation'
elif page_selection == "Fertilizer Recommendation":
    st.session_state['page'] = 'fertilizer_recommendation'
elif page_selection == "Crop Yield Prediction":
    st.session_state['page'] = 'crop_yield_prediction'
else:
    st.session_state['page'] = 'home'


# Define language options
languages = {
    'English': 'en',
    'Polish': 'pl',
    'German': 'de'
}

# Create a dropdown for language selection
selected_language = st.selectbox('Select language:', list(languages.keys()))

# Display translated text based on selected language
if selected_language == 'English':
    st.write("Hello! Welcome to the English version.")
elif selected_language == 'Polish':
    st.write("Witaj! Witamy w polskiej wersji.")
elif selected_language == 'German':
    st.write("Hallo! Willkommen zur deutschen Version.")

# Home page
if st.session_state['page'] == 'home':
    st.write("# AI-ML Decision System for Effective Farming")
    st.image("https://cdni.iconscout.com/illustration/premium/thumb/indian-farmer-showing-mobile-2773411-2319316.png", use_column_width=True)

# Crop recommendation page
if 'page' in st.session_state and st.session_state['page'] == 'crop_recommendation':

    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    [data-testid="stSidebar"] {{
    background-color:#8F9779; /* unchanged */

    }}
    [data-testid="stHeader"] {{
    background-color:#90EE90; /* changed to #90EE90 */
    }}
    [data-testid="stToolbar"] {{
    background-color:#90EE90; /* changed to #90EE90 */
    }}
    
    </style>
    """
    st.markdown(page_bg,unsafe_allow_html=True)

    def load_bootstrap():
            return st.markdown("""<link rel="stylesheet" 
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
            integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
            crossorigin="anonymous">""", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: black;'>Crop Recommendation System</h1>", unsafe_allow_html=True)

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')

    df = pd.read_csv('Crop_recommendation.csv')

    rdf_clf = joblib.load('final_rdf_clf.pkl')

    X = df.drop('label', axis = 1)
    y = df['label']

    df_desc = pd.read_csv('Crop_Desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')

    st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model:</h5>", unsafe_allow_html=True)

    importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
    importance['Importance(%)'] = importance['Importance(%)'] * 100

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')
    with coly:
        color_discrete_sequence = '#609cd4'
        fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
        fig.update_traces(marker_color="#8C564B")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width= True)

    st.text("Now insert the values and the system will predict the best crop to plant.")
    st.text("In the (?) marks you can get some help about each feature.")

    col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,4,1,4,1,1], gap = 'medium')

    with col3:
        n_input = st.number_input('Insert N (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Nitrogen density (kg/ha) from 0 to 140.')
        p_input = st.number_input('Insert P (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Phosphorus density (kg/ha) from 5 to 145.')
        k_input = st.number_input('Insert K (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Potassium density (kg/ha) from 5 to 205.')
        temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value= 9., max_value= 49., step = 1., format="%.2f", help = 'Insert here the Avg Temperature (ºC) from 9 to 43.')

    with col5:
        hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value= 14., max_value= 100., step = 1., format="%.2f", help = 'Insert here the Avg Humidity (%) from 15 to 99.')
        ph_input = st.number_input('Insert pH value:', min_value= 3.6, max_value= 9.9, step = 0.1, format="%.2f", help = 'Insert here the pH from 3.6 to 9.9')
        rain_input = st.number_input('Insert Avg Rainfall (mm) value:', min_value= 30.0, max_value= 2700.0, step = 0.1, format="%.2f", help = 'Insert here the Avg Rainfall (mm) from 21 to 2700')

    predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input]]

    with col5:
        predict_btn = st.button('Get Your Recommendation!')

    cola,colb,colc = st.columns([2,10,2])
    if predict_btn:
        rdf_predicted_value = rdf_clf.predict(predict_inputs)
        #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
        with colb:
            st.markdown(f"<h3 style='text-align: center;'>Best Crop to Plant: {rdf_predicted_value[0]}.</h3>", 
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns([9,4,9])
        with col2:
            df_desc = df_desc.astype({'label':str,'image':str})
            df_desc['label'] = df_desc['label'].str.strip()
            df_desc['image'] = df_desc['image'].str.strip()

            df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
            df_image = df_pred_image['image'].item()

            st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)
        
        # Text-to-speech functionality
        tts_text = f"The best crop to plant is {rdf_predicted_value[0]}"
        components.iframe(f"https://responsivevoice.org/text-to-speech-demo/?text={tts_text}&voice=UK%20English%20Male", width=400, height=300)

# Fertilizer recommendation page
if 'page' in st.session_state and st.session_state['page'] == 'fertilizer_recommendation':
    # Add your fertilizer recommendation code here
    Fertilizer = pd.read_csv('fertilizer_data.csv', index_col='Original')
    df_desc = pd.read_csv('fert_desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    [data-testid="stSidebar"] {{
    background-color:#8F9779; /* unchanged */

    }}
    [data-testid="stHeader"] {{
    background-color:#90EE90; /* changed to #90EE90 */
    }}
    [data-testid="stToolbar"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    </style>
    """
    st.markdown(page_bg,unsafe_allow_html=True)

    def load_bootstrap():
            return st.markdown("""<link rel="stylesheet" 
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
            integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
            crossorigin="anonymous">""", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Fertilizer Recommendation System</h1>", unsafe_allow_html=True)

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')

    df = pd.read_csv('fert_dataset.csv')

    rdf_clf = joblib.load('ferti_rdf_clf.pkl')

    X = df.drop('Fertilizer Name', axis = 1)
    y = df['Fertilizer Name']

    
    st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model:</h5>", unsafe_allow_html=True)

    importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
    importance['Importance(%)'] = importance['Importance(%)'] * 100

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')
    with coly:
        color_discrete_sequence = '#609cd4'
        fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
        fig.update_traces(marker_color="#8C564B")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width= True)

    st.text("Now insert the values and the system will predict the best crop to plant.")
    st.text("In the (?) marks you can get some help about each feature.")

    

    

    st.text("Now insert the values and the system will predict the fertilizer.")
    st.text("In the (?) marks you can get some help about each feature.")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,4,1,4,1,1,1], gap = 'medium')

    with col3:
        n_input = st.number_input('Insert N (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Nitrogen density (kg/ha) from 0 to 140.')
        p_input = st.number_input('Insert P (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Phosphorus density (kg/ha) from 5 to 145.')
        k_input = st.number_input('Insert K (kg/ha) value:', min_value= 0, max_value= 300, help = 'Insert here the Potassium density (kg/ha) from 5 to 205.')
        temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value= 9., max_value= 49., step = 1., format="%.2f", help = 'Insert here the Avg Temperature (ºC) from 9 to 43.')

    with col5:
        hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value= 14., max_value= 100., step = 1., format="%.2f", help = 'Insert here the Avg Humidity (%) from 15 to 99.')
        moisture_input = st.number_input('Insert Soil Moisture',  help = 'Insert here the soil moisture ')
        soil_input = st.selectbox('Select Soil Type:', df['Soil Type'].unique(), format_func=lambda x: x)
        soil_type_input = np.where(df['Soil Type'].unique() == soil_input)[0][0]

        crop_input = st.selectbox('Select Crop Type:', df['Crop Type'].unique(), format_func=lambda x: x)
        crop_type_input = np.where(df['Crop Type'].unique() == crop_input)[0][0]

    predict_inputs = [[temp_input,hum_input,moisture_input,soil_type_input,crop_type_input,n_input,p_input,k_input,]]

    with col5:
        predict_btn = st.button('Get Your Recommendation!')

    cola,colb,colc = st.columns([2,10,2])
    if predict_btn:
        rdf_predicted_value = rdf_clf.predict(predict_inputs)
        encoded_predicted_fertilizer = rdf_predicted_value[0]

        # Retrieve the corresponding fertilizer name from the Fertilizer DataFrame
        predicted_fertilizer_name = Fertilizer.iloc[encoded_predicted_fertilizer].name
        #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
        with colb:
            st.markdown(f"<h3 style='text-align: center;'>Best Fertilizer to use : {predicted_fertilizer_name}.</h3>", 
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns([9,4,9])
        with col2:
            df_desc = df_desc.astype({'label':str,'image':str})
            df_desc['label'] = df_desc['label'].str.strip()
            df_desc['image'] = df_desc['image'].str.strip()
            
            
            
            
            
            l = []
            l.append(predicted_fertilizer_name)
            l[0] += '.'
            
            df_pred_image = df_desc[df_desc['label'].isin(l)]
            
            
            df_image = df_pred_image['image'].item()
            st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)
    

# Crop yield prediction page
if 'page' in st.session_state and st.session_state['page'] == 'crop_yield_prediction':
    # Add your crop yield prediction code here
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    [data-testid="stSidebar"] {{
    background-color:#8F9779; /* unchanged */

    }}
    [data-testid="stHeader"] {{
    background-color:#90EE90; /* changed to #90EE90 */
    }}
    [data-testid="stToolbar"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    </style>
    """
    st.markdown(page_bg,unsafe_allow_html=True)

    def load_bootstrap():
            return st.markdown("""<link rel="stylesheet" 
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
            integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
            crossorigin="anonymous">""", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Crop Yield Prediction</h1>", unsafe_allow_html=True)

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')

    df = pd.read_csv('crop_production.csv')

    rdf_clf = joblib.load('yield_rdf_clf.pkl')

    X = df.drop(["Production"], axis=1)
   
    

    st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model:</h5>", unsafe_allow_html=True)

    importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
    importance['Importance(%)'] = importance['Importance(%)'] * 100

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')
    with coly:
        color_discrete_sequence = '#609cd4'
        fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
        fig.update_traces(marker_color="#8C564B")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width= True)

    st.text("Now insert the values and the system will predict the yield.")
    st.text("In the (?) marks you can get some help about each feature.")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,4,1,4,1,1,1], gap = 'medium')

    with col3:
        state_input = st.selectbox('Select State:', df['State_Name'].unique(), format_func=lambda x: x)
        state_input = np.where(df['State_Name'].unique() == state_input)[0][0]
        district_input = st.selectbox('Select District:', df['District_Name'].unique(), format_func=lambda x: x)
        district_input = np.where(df['District_Name'].unique() == district_input)[0][0]
        crop_year_input = st.number_input(' Enter Crop Year value:', min_value= 1998,  help = 'Insert here the year')
        season_type_input = st.selectbox('Select Season:', df['Season'].unique(), format_func=lambda x: x)
        season_input = np.where(df['Season'].unique() == season_type_input)[0][0]
        

    with col5:
        crop_input = st.selectbox('Select Crop Type:', df['Crop'].unique(), format_func=lambda x: x)
        crop_type_input = np.where(df['Crop'].unique() == crop_input)[0][0]
        area_input = st.number_input('Enter Area:', min_value= 0,  help = 'Insert here the area .')
        
    predict_inputs = [[state_input,district_input,crop_year_input,season_input,crop_type_input,area_input]]

    with col5:
        predict_btn = st.button('Get Your Prediction!')

    cola,colb,colc = st.columns([2,10,2])
    if predict_btn:
        rdf_predicted_value = rdf_clf.predict(predict_inputs)
        
        with colb:
            st.markdown(f"<h3 style='text-align: center;'>Yield Prediction : {rdf_predicted_value[0]}.</h3>", 
            unsafe_allow_html=True)
        

