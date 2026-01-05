import streamlit as st

from wacc_calculator import WaccCalculator

st.title("🎈 SSP-linked Cost of Capital Scenarios")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


wacc_calculator = WaccCalculator(GDP_data="GDP_Historical.csv", SSP_data="SSP_OECD_ENV.csv", CRP="Collated_CRP_CDS.xlsx", CDS="Collated_CRP_CDS.xlsx")
wacc_calculator.calculate_wacc_scenarios()