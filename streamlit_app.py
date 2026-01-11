import streamlit as st
from wacc_calculator import WaccCalculator
import altair as alt
import matplotlib.pyplot as plt


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

def plot_ssp_comparison(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Overall Cost of Capital:Q', title='Overall Cost of Capital (%)'),
            color=alt.Color('Scenario:N', title='Scenario'),
            column=alt.Column('Country Name:N', title='Country Name'),
            tooltip=['Year', 'Scenario', 'Overall Cost of Capital']
        )
        .properties(width=700)
    )

    st.write(chart)

def plot_ssp_comparison_matplotlib(df):
    plt.figure(figsize=(10, 6))
    for scenario in df['Scenario'].unique():
        scenario_data = df[df['Scenario'] == scenario]
        plt.plot(scenario_data['Year'], scenario_data['Overall Cost of Capital'], marker='o', label=scenario)

    plt.title('Overall Cost of Capital by Scenario')
    plt.xlabel('Year')
    plt.ylabel('Overall Cost of Capital (%)')
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig('ssp_comparison_matplotlib.png')  # Save the figure as a PNG file
    plt.show()



st.title("🎈 SSP-linked Cost of Capital Scenarios")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


wacc_calculator = WaccCalculator(GDP_data="GDP_Historical.csv", SSP_data="SSP_OECD_ENV.csv", CRP="Collated_CRP_CDS.xlsx", 
                                 CDS="Collated_CRP_CDS.xlsx", tax_data="CORPORATE_TAX_DATA.csv", debt_data="IMF_Government_Debt.csv", 
                                 inflation_data = "IMF_Inflation_Rates.csv", deficit_data="IMF_Overall_Balance.csv", country_coding="Country_Coding.csv")
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
scenario_comparison = scenario.loc[(scenario["Country Name"].isin(["EMDE Mean", "Advanced Mean"]))  & (scenario["Technology"] == "Clean")]
plot_ssp_comparison(scenario_comparison)
plot_ssp_comparison_matplotlib(scenario_comparison)

