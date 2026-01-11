import streamlit as st
from wacc_calculator import WaccCalculator
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


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


def plot_region_boxplots_by_ssp_matplotlib(df, year=2050, regions=None, technology='Clean'):
    """Create horizontal grouped boxplots by Region showing the distribution of
    'Overall Cost of Capital' for each SSP in a specified year.

    Args:
        df (pd.DataFrame): DataFrame containing at minimum ['Year','Region','Scenario','Country Name','Overall Cost of Capital','Technology']
        year (int): Year to plot (default 2050)
        regions (list[str] | None): Ordered list of regions to include. If None, a sensible default is used.
        technology (str | None): If provided, filter by Technology (default 'Clean')
    """
    # Default region order
    if regions is None:
        regions = [
            'North America',
            'Latin America and the Caribbean',
            'Africa',
            'Western Europe',
            'Eastern Europe',
            'Asia',
            'Oceania'
        ]

    # Basic checks
    if 'Region' not in df.columns:
        st.warning("No 'Region' column found in DataFrame. Ensure aggregates have been merged.")
        return

    plot_df = df.copy()
    plot_df['Country Name'] = plot_df['Country Name'].astype(str).str.strip()
    plot_df['Region'] = plot_df['Region'].astype(str).str.strip()

    # Ensure Year is numeric where possible
    try:
        plot_df['Year'] = plot_df['Year'].astype(int)
    except Exception:
        pass

    # Filter
    plot_df = plot_df[plot_df['Year'] == int(year)]
    if technology is not None:
        plot_df = plot_df[plot_df['Technology'] == technology]

    # Exclude aggregates so boxplots show country ranges
    plot_df = plot_df[~plot_df['Country Name'].str.contains('Mean', na=False)]

    # Filter to regions and SSPs
    plot_df = plot_df[plot_df['Region'].isin(regions)]
    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    plot_df = plot_df[plot_df['Scenario'].isin(ssp_list)]

    if plot_df.empty:
        st.warning(f"No data available for year {year} with the given filters.")
        return

    n_regions = len(regions)
    y = np.arange(n_regions)
    n_ssp = len(ssp_list)
    width = 0.13

    colors = plt.get_cmap('Set2')(np.linspace(0, 1, n_ssp))

    fig, ax = plt.subplots(figsize=(10, max(6, n_regions * 0.6)))

    # Draw boxplots for each SSP with horizontal orientation and slight offsets
    for j, ssp in enumerate(ssp_list):
        data_j = [
            plot_df[(plot_df['Region'] == r) & (plot_df['Scenario'] == ssp)]['Overall Cost of Capital'].dropna().values
            for r in regions
        ]
        data_j = [d if len(d) > 0 else np.array([np.nan]) for d in data_j]
        positions = y + (j - (n_ssp - 1) / 2) * width

        bp = ax.boxplot(
            data_j,
            positions=positions,
            widths=width * 0.9,
            vert=False,
            patch_artist=True,
            manage_ticks=False
        )
        for patch in bp['boxes']:
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(regions)
    ax.set_xlabel('Overall Cost of Capital (%)')
    ax.set_title(f'Overall Cost of Capital by Region (Year {year})')

    legend_patches = [Patch(facecolor=colors[i], label=ssp_list[i]) for i in range(n_ssp)]
    ax.legend(handles=legend_patches, title='SSP', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)

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

# Regional boxplots (easy to change year)
years = sorted(scenario['Year'].unique())
default_index = years.index(2050) if 2050 in years else 0
year_choice = st.selectbox('Year (for regional boxplots)', years, index=default_index)
plot_region_boxplots_by_ssp_matplotlib(scenario, year=year_choice, technology='Clean')

