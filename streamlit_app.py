import streamlit as st
from wacc_calculator import WaccCalculator
import altair as alt

def plot_comparison_chart_equity(df):
   # Melt dataframe
    data_melted = df.melt(id_vars="Year", var_name="Factor", value_name="Value")

    # Set order
    category_order = ['Risk Free Rate', 'Country Risk Premium', 'Equity Risk Premium', 'Technology Risk Premium']

    # Create chart
    chart = alt.Chart(data_melted).mark_bar().encode(
        x=alt.X('sum(Value):Q', stack='zero', title='Weighted Average Cost of Capital (%)'),
        y=alt.Y('Year:O', title='Country'),  # Sort countries by total value descending
        color=alt.Color('Factor:N', title='Factor'),
        order=alt.Order('Factor:O', sort="ascending"),  # Color bars by category
).properties(width=700)
    st.write(chart)

def plot_comparison_chart_debt(df):
   # Melt dataframe
    data_melted = df.melt(id_vars="Year", var_name="Factor", value_name="Value")

    # Set order
    category_order = ['Risk Free Rate', 'Country Default Spread', 'Lenders Margin', 'Technology Risk Premium']

    # Create chart
    chart = alt.Chart(data_melted).mark_bar().encode(
        x=alt.X('sum(Value):Q', stack='zero', title='Weighted Average Cost of Capital (%)'),
        y=alt.Y('Year:O', title='Country'),  # Sort countries by total value descending
        color=alt.Color('Factor:N', title='Factor'),
        order=alt.Order('Factor:O', sort="ascending"),  # Color bars by category
).properties(width=700)
    st.write(chart)



st.title("🎈 SSP-linked Cost of Capital Scenarios")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


wacc_calculator = WaccCalculator(GDP_data="GDP_Historical.csv", SSP_data="SSP_OECD_ENV.csv", CRP="Collated_CRP_CDS.xlsx", 
                                 CDS="Collated_CRP_CDS.xlsx", tax_data="CORPORATE_TAX_DATA.csv")
scenario = wacc_calculator.calculate_wacc_scenarios()
#central_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity=None)
#high_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity="High")


SSP = st.selectbox(
        "SSP", ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5"), 
         index=0, key="SSP", placeholder="Select SSP...")
country = st.selectbox(
        "Displayed Country", scenario["Country Name"].unique(), 
         index=25, placeholder="Select Country...", key="Country")

# Select data based on input
selected_data = scenario.loc[(scenario["Scenario"] == SSP) & (scenario["Country Name"] == country)  & (scenario["Technology"] == "Clean")]
plot_comparison_chart_equity(selected_data[["Year", "Risk Free Rate", "Country Risk Premium", "Equity Risk Premium", "Technology Risk Premium"]])
plot_comparison_chart_debt(selected_data[["Year", "Risk Free Rate", "Country Default Spread", "Lenders Margin", "Technology Risk Premium"]])