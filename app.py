import streamlit as st
import plotly.express as px
import numpy as np
import pickle
from predict import predict_flower
import time


st.title("My Iris Predictor App")
st.header("It predicts Iris flower types")
st.subheader("Cool, huh?")

df_iris = px.data.iris()

# must save as variable 
hist_sepal = px.histogram(df_iris, x='sepal_length')
hist_sepal    


# 
df = px.data.gapminder()

fig = px.choropleth(
    df,
    locations="iso_alpha",
    color="lifeExp",
    hover_name="country",
    animation_frame="year",  # animation
    range_color=[20, 80],
)
fig

show_df = st.checkbox("Do you want to see the data?")
# show_df  # boolean

# st.write(type(show_df))

if show_df:  # evaluates to true if checked
    df_iris

s_l = st.number_input("Sepal Length in cm", 0, 100)
s_w = st.number_input("Sepal Width in cm", 0, 100)
p_l = st.number_input("Petal Length in cm", 0, 100)
p_w = st.number_input("Petal Width in cm", 0, 100)

user_input = np.array([s_l, s_w, p_l, p_w])
user_input

# load model
with open('saved-iris-model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# make prediction
with st.spinner('Predicting...'):
    time.sleep(5)
    prediction = predict_flower(classifier, user_input)

st.header(f"The model predicts: {prediction[0]}")


col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    if prediction[0]=='versicolor':
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")
    else:
        st.image("https://static.streamlit.io/examples/dog.jpg")

st.balloons()