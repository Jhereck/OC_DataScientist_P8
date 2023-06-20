import streamlit as st
import pandas as pd
import pickle as pk
import plotly.express as px

# Load datas

datas = pd.read_csv("./datas/test_feature_engineering.csv")
datas.drop("Unnamed: 0", axis=1, inplace=True)


# Load model
model = pk.load(open("./model/model.pkl", "rb"))
explainer = pk.load(open("./model/shap_explainer.pkl", "rb"))


shap_values = explainer(datas)

st.title("Prédictions sur les potentiels bons et mauvais payeurs")


st.number_input(
    label="Numéro du client",
    key="num_client",
    step=1,
    min_value=0,
    max_value=len(datas),
    value=0,
)


predict = model.predict_proba([datas.iloc[st.session_state.num_client]])
if predict[:, 1] >= 0.4669:
    f"Le client {st.session_state.num_client} est un potentiel mauvais payeur ({round(predict[0,1], 3)*100} %)"


f"Le client {st.session_state.num_client} est un potentiel bon payeur ({round(predict[0,0], 3)*100} %)"

d = {
    "Label": ["Probabilité bon payeur", "Probabilité mauvais payeur"],
    "Valeurs": [predict[0, 0], predict[0, 1]],
}

s = {
    "Feature": explainer.data_feature_names,
    "Value": shap_values.values[st.session_state.num_client],
}

chart_data = pd.DataFrame(data=d)

st.write(chart_data)

pie = [predict[0].tolist()[0], predict[0].tolist()[1]]


pie_fig = px.pie(chart_data, values=pie, names="Label")
st.plotly_chart(pie_fig, use_container_width=True)

numb_shap = st.slider(
    "Selectionnez le nombre de features locales qui ont le plus d'influence à afficher",
    0,
    len(datas.columns),
    (10),
    step=1,
)
bar_data = pd.DataFrame(data=s)
bar_data.sort_values(by="Value", ascending=False, inplace=True)


bar_fig = px.bar(bar_data[:numb_shap], x="Feature", y="Value")
st.plotly_chart(bar_fig, use_container_width=True)


thersh_shap = st.slider(
    "Ajustez le seuil d'importance positive ou négative pour afficher les features locales correspondantes",
    bar_data["Value"].min(),
    bar_data["Value"].max(),
    (0.00),
    step=0.01,
)

if thersh_shap >= 0.00:
    bar_data_filtred = bar_data.loc[bar_data["Value"] >= thersh_shap]
else:
    bar_data_filtred = bar_data.loc[bar_data["Value"] <= thersh_shap]

bar_fig = px.bar(bar_data_filtred, x="Feature", y="Value")
st.plotly_chart(bar_fig, use_container_width=True)
