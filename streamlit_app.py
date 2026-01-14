import streamlit as st
from wacc_calculator import WaccCalculator
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as _pd
from pathlib import Path as _Path
from matplotlib.lines import Line2D
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Consistent color mapping for SSPs across matplotlib plots
SSP_COLOR_MAP = {
    'SSP1': '#2ca02c',  # green
    'SSP2': '#ff7f0e',  # orange
    'SSP3': '#1f77b4',  # blue
    'SSP4': '#9467bd',  # purple
    'SSP5': '#d62728',  # red
}

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

def plot_ssp_comparison_matplotlib(df, figsize=(12, 6), save_path='ssp_comparison_matplotlib.png', show=True):
    """Plot overall cost of capital by technology (Clean vs Fossil) for EMDEs and Advanced Economies.

    - Creates two subplots side-by-side: left = Clean, right = Fossil
    - Scenario -> color
    - Country Name (EMDEs / Advanced Economies) -> line style
    Legend is placed below the two subplots (shared).
    Returns (fig, axes).
    """

    from matplotlib.lines import Line2D

    # Technologies and countries of interest
    technologies = ['Clean', 'Fossil']
    countries = ['EMDEs', 'Advanced Economies']

    # Prepare figure with two subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    # Determine scenarios and style pools
    scenarios = sorted(df['Scenario'].unique())
    linestyles = ['-', '--', '-.', ':']
    tab_colors = plt.get_cmap('tab10')(range(10))

    # Assign a linestyle to each country of interest
    country_styles = {country: linestyles[i % len(linestyles)] for i, country in enumerate(countries)}

    # Plot each subplot for each technology
    for ax, tech in zip(axes, technologies):
        for i, scenario in enumerate(scenarios):
            color = SSP_COLOR_MAP.get(scenario, tab_colors[i % len(tab_colors)])
            for country in countries:
                data = df[(df['Scenario'] == scenario) & (df['Country Name'] == country) & (df.get('Technology') == tech)]
                if data.empty:
                    continue
                ls = country_styles[country]
                ax.plot(data['Year'], data['Overall Cost of Capital'], linestyle=ls, color=color, linewidth=1.5)
        ax.set_title(f"{tech}")
        ax.set_xlabel('Year')

    # Y label on the left subplot
    axes[0].set_ylabel('Mean Cost of Capital (%, nominal)')

    # Create shared legend (scenarios colors + country linestyles) and place it below the subplots
    scenario_handles = [Line2D([0], [0], color=SSP_COLOR_MAP.get(scenarios[i], tab_colors[i % len(tab_colors)]), lw=2) for i in range(len(scenarios))]
    country_handles = [Line2D([0], [0], color='k', lw=1.5, linestyle=country_styles[c]) for c in countries]

    handles = scenario_handles + country_handles
    labels = scenarios + countries

    # Place combined legend centered below plots
    fig.legend(handles, labels, loc='lower center', ncol=max(3, len(labels)), bbox_to_anchor=(0.5, -0.05), frameon=False)

    # Adjust layout to make room for legend
    fig.subplots_adjust(bottom=0.1, wspace=0.15)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


def plot_ssp_comparison_range_matplotlib(scenario, low_scenario, high_scenario, technology='Clean', include_both=False, figsize=(12, 10), save_path='ssp_comparison_range_matplotlib.png', show=True):
    """Plot five subplots (one per SSP) showing EMDEs and Advanced Economies means with low/high ranges.

    Parameters
    ----------
    scenario, low_scenario, high_scenario : pd.DataFrame
        DataFrames containing columns ['Scenario','Country Name','Year','Overall Cost of Capital','Technology'].
    technology : str
        Primary technology to plot when include_both is False (default 'Clean').
    include_both : bool
        If True, plot both 'Clean' and 'Fossil' on the same axes for each SSP.
    Returns
    -------
    (fig, axes)
    """

    from matplotlib.patches import Patch

    countries = ['EMDEs', 'Advanced Economies']
    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']

    # Grid: 3 rows x 2 cols, last axis spans both columns (centered)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])]

    tab_colors = plt.get_cmap('tab10')(range(10))
    country_styles = {'EMDEs': '--', 'Advanced Economies': '-'}

    # Loop over SSPs and plot into subplots
    for idx, ssp in enumerate(ssp_list):
        ax = axes[idx]
        ax.set_title(ssp)
        ax.set_xlabel('Year')
        color = SSP_COLOR_MAP.get(ssp, tab_colors[idx % len(tab_colors)])

        techs = ['Clean', 'Fossil'] if include_both else [technology]

        for tech in techs:
            for country in countries:
                # Filter frames
                central = scenario[(scenario['Scenario'] == ssp) & (scenario['Country Name'] == country) & (scenario.get('Technology') == tech)].copy()
                low = low_scenario[(low_scenario['Scenario'] == ssp) & (low_scenario['Country Name'] == country) & (low_scenario.get('Technology') == tech)].copy()
                high = high_scenario[(high_scenario['Scenario'] == ssp) & (high_scenario['Country Name'] == country) & (high_scenario.get('Technology') == tech)].copy()

                if central.empty:
                    continue

                # Coerce Year to numeric and ensure ordering
                central['Year'] = _pd.to_numeric(central['Year'], errors='coerce')
                central = central.dropna(subset=['Year'])
                central['Year'] = central['Year'].astype(int)
                years = list(central['Year'].unique())

                # Prepare low/high and central series aggregated by Year
                low['Year'] = _pd.to_numeric(low['Year'], errors='coerce')
                high['Year'] = _pd.to_numeric(high['Year'], errors='coerce')

                low_series = low.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)
                high_series = high.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)
                central_series = central.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)

                # Numeric coercion and interpolation
                low_series = _pd.to_numeric(low_series, errors='coerce').astype(float)
                high_series = _pd.to_numeric(high_series, errors='coerce').astype(float)
                central_series = _pd.to_numeric(central_series, errors='coerce').astype(float)

                low_series = low_series.interpolate(limit_direction='both').fillna(central_series)
                high_series = high_series.interpolate(limit_direction='both').fillna(central_series)

                years_arr = np.array(years, dtype=float)
                low_arr = low_series.to_numpy(dtype=float)
                high_arr = high_series.to_numpy(dtype=float)
                central_arr = central_series.to_numpy(dtype=float)

                # Plot envelope and central line
                ax.fill_between(years_arr, low_arr, high_arr, color=color, alpha=0.12, linewidth=0)

                # Distinguish Fossil vs Clean by marker (Fossil gets markers)
                if tech.lower() == 'fossil':
                    marker = 'o'
                    markersize = 4
                    ax.plot(years_arr, central_arr, color=color, lw=1.5, linestyle=country_styles[country], marker=marker, markersize=markersize, markerfacecolor=color, markeredgecolor='k')
                else:
                    ax.plot(years_arr, central_arr, color=color, lw=2, linestyle=country_styles[country])

        ax.grid(alpha=0.35, linestyle='--')

    axes[0].set_ylabel('Mean Cost of Capital (%, nominal)')

    # Legend: country line styles + fossil marker + envelope patch
    country_handles = [Line2D([0], [0], color='k', lw=1.5, linestyle=country_styles[c], label=c) for c in countries]
    fossil_handle = Line2D([0], [0], color='k', lw=1.5, marker='o', label='Fossil', markerfacecolor='k', markersize=5)
    envelope_patch = Patch(facecolor='gray', alpha=0.12, label='Low / High range')

    handles = country_handles + ([fossil_handle] if include_both else []) + [envelope_patch]
    labels = [h.get_label() for h in handles]

    fig.legend(handles, labels, loc='lower center', ncol=max(3, len(labels)), bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25)

    # Center the bottom subplot so it has the same width as the top columns
    try:
        left_pos = axes[0].get_position()
        right_pos = axes[1].get_position()
        column_width = left_pos.width
        center_x = (left_pos.x0 + right_pos.x0 + right_pos.width) / 2.0
        bottom_pos = axes[4].get_position()
        new_x0 = center_x - column_width / 2.0
        axes[4].set_position([new_x0, bottom_pos.y0, column_width, bottom_pos.height])
    except Exception:
        # If anything goes wrong, don't block plotting
        pass

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


def plot_dgs10_line(filepath=None, start_date=None, end_date=None, rolling=None, figsize=(10, 6), ax=None, show=True, save_path=None):
    """Plot the US 10-year Treasury yield from the DGS10 CSV.

    Parameters
    ----------
    filepath : str | pathlib.Path | None
        Path to the DGS10 CSV. If None, uses DATA/DGS10.csv next to this file.
    start_date, end_date : str | datetime | None
        Optional date bounds to filter the data (inclusive).
    rolling : int | None
        If provided and >1, plots a rolling mean with this window size.
    figsize : tuple
        Figure size when a new figure is created.
    ax : matplotlib.axes.Axes | None
        Axis to plot on. If None, a new figure/axis pair is created.
    show : bool
        Whether to call ``plt.show()`` after plotting.
    save_path : str | None
        If provided, the figure will be saved to this path.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axis objects.
    """


    path = _Path(filepath) if filepath is not None else _Path(__file__).resolve().parent / "DATA" / "DGS10.csv"
    df = _pd.read_csv(path, parse_dates=["observation_date"])  # expects columns observation_date, DGS10
    df = df.sort_values("observation_date").dropna(subset=["DGS10"])  # drop missing yields

    # Apply optional date filtering
    if start_date is not None:
        start = _pd.to_datetime(start_date)
        df = df[df["observation_date"] >= start]
    if end_date is not None:
        end = _pd.to_datetime(end_date)
        df = df[df["observation_date"] <= end]

    x = df["observation_date"]
    y = _pd.to_numeric(df["DGS10"], errors="coerce")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(x, y, label="10 year U.S. Treasury Yield", color="C0", linewidth=1)
    if rolling and isinstance(rolling, int) and rolling > 1:
        ax.plot(x, y.rolling(window=rolling).mean(), label=f"{rolling}-day MA", color="C1", linewidth=1.25)

    ax.set_xlabel("Date")
    ax.set_ylabel("10-Year Treasury Yield (%, nominal)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


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

    # Duplicate filtered dataframe and set Region to the country's income grouping so we can plot income-group boxplots
    income_categories = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    combined_df = plot_df.copy()
    if 'wb_income_group' in plot_df.columns:
        income_df = plot_df[plot_df['wb_income_group'].notna()].copy()
        income_df['Region'] = income_df['wb_income_group']
        # Place income group rows first so they appear at the top of the plot
        combined_df = _pd.concat([income_df, combined_df], ignore_index=True)

    # Final regions ordering: put WB income categories at the top (in logical order), then the remaining geographic regions
    final_regions = []
    for ic in income_categories:
        if ic in combined_df['Region'].unique():
            final_regions.append(ic)
    # Add geographic regions while avoiding duplicates
    final_regions += [r for r in regions if r not in final_regions]

    n_regions = len(final_regions)
    y = np.arange(n_regions)
    n_ssp = len(ssp_list)
    width = 0.13

    # Use consistent SSP colors (SSP1 green, SSP5 red, others distinct)
    colors = [SSP_COLOR_MAP.get(ssp, plt.get_cmap('Set2')(i / max(1, n_ssp-1))) for i, ssp in enumerate(ssp_list)]

    fig, ax = plt.subplots(figsize=(10, max(6, n_regions * 0.6)))

    # Draw boxplots for each SSP with horizontal orientation and slight offsets
    for j, ssp in enumerate(ssp_list):
        data_j = [
            combined_df[(combined_df['Region'] == r) & (combined_df['Scenario'] == ssp)]['Overall Cost of Capital'].dropna().values
            for r in final_regions
        ]
        data_j = [d if len(d) > 0 else np.array([np.nan]) for d in data_j]
        positions = y + (j - (n_ssp - 1) / 2) * width

        bp = ax.boxplot(
            data_j,
            positions=positions,
            widths=width * 0.9,
            vert=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops={'color': 'black', 'linewidth': 1.2}
        )
        for patch in bp['boxes']:
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(final_regions)
    ax.set_xlabel(f'Overall Cost of Capital (%, {year}, {technology})')

    legend_patches = [Patch(facecolor=colors[i], label=ssp_list[i]) for i in range(n_ssp)]
    ax.legend(handles=legend_patches, title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    st.pyplot(fig)
    fig.savefig("./PLOTS/boxplots_region_aggregates.png", bbox_inches="tight", dpi=300)


def plot_wacc_world_heatmap(selected_scenario, year_choice, technology='Clean', figsize=(20, 12), save_path='wacc_world_heatmap.png', show=True):
    """Plot five subplots (one per SSP) showing world map heatmaps of Overall Cost of Capital by country.

    Parameters
    ----------
    selected_scenario : pd.DataFrame
        DataFrame containing columns ['Scenario','Country code','Year','Overall Cost of Capital','Technology'].
    year_choice : int
        Year to filter the data for.
    technology : str
        Technology to plot ('Clean' or 'Fossil'). Default 'Clean'.
    figsize : tuple
        Figure size for the output. Default (20, 12).
    save_path : str | None
        Path to save the figure. Default 'wacc_world_heatmap.png'.
    show : bool
        Whether to display the figure. Default True.

    Returns
    -------
    (fig, axes)
        The matplotlib figure and axes objects.
    """
    
    # Load world map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    
    # SSP list
    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    
    # Filter data by year and technology
    plot_df = selected_scenario[
        (selected_scenario['Year'] == year_choice) & 
        (selected_scenario['Technology'] == technology)
    ].copy()
    
    if plot_df.empty:
        raise ValueError(f"No data available for year {year_choice} and technology {technology}")
    
    # Merge country codes to ISO3 codes for mapping to geopandas
    # Try to use 'Country code' if available, otherwise use 'iso_a3' from the data
    if 'Country code' not in plot_df.columns and 'iso_a3' not in plot_df.columns:
        st.warning("No 'Country code' or 'iso_a3' column found in data.")
        return None, None
    
    # Use Country code as the merge key
    code_column = 'Country code' if 'Country code' in plot_df.columns else 'iso_a3'
    
    # Prepare data: group by Scenario and Country code, taking the mean of Overall Cost of Capital
    plot_df_agg = plot_df.groupby(['Scenario', code_column])['Overall Cost of Capital'].mean().reset_index()
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Calculate min and max for consistent colorbar across all subplots
    vmin = plot_df_agg['Overall Cost of Capital'].min()
    vmax = plot_df_agg['Overall Cost of Capital'].max()
    cmap = plt.cm.RdYlGn_r  # Red=high, Yellow=medium, Green=low
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot each SSP
    for idx, ssp in enumerate(ssp_list):
        ax = axes[idx]
        
        # Filter data for this SSP
        ssp_data = plot_df_agg[plot_df_agg['Scenario'] == ssp].copy()
        
        # Merge with world map
        world_plot = world.merge(
            ssp_data,
            left_on='iso_a3',
            right_on=code_column,
            how='left'
        )
        
        # Plot base map (light gray for countries without data)
        world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
        
        # Plot data
        world_plot_with_data = world_plot[world_plot['Overall Cost of Capital'].notna()]
        world_plot_with_data.plot(
            ax=ax,
            column='Overall Cost of Capital',
            cmap=cmap,
            norm=norm,
            edgecolor='#333333',
            linewidth=0.3,
            legend=False
        )
        
        ax.set_title(f'{ssp} - {technology} ({year_choice})', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Remove the 6th subplot (we only need 5 for SSPs)
    fig.delaxes(axes[5])
    
    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Overall Cost of Capital (%)', rotation=270, labelpad=20, fontsize=10)
    
    fig.suptitle(f'World Map Heatmap: Cost of Capital by SSP ({technology}, {year_choice})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    
    return fig, axes


st.title("🎈 SSP-linked Cost of Capital Scenarios")
st.write(
    "Dashboard work in progress: enables easy visualisation of the results and underlying model framework"
)


wacc_calculator = WaccCalculator(GDP_data="GDP_Historical.csv", SSP_data="SSP_OECD_ENV.csv", CRP="Collated_CRP_CDS.xlsx", 
                                 CDS="Collated_CRP_CDS.xlsx", tax_data="CORPORATE_TAX_DATA.csv", debt_data="IMF_Government_Debt.csv", 
                                 inflation_data = "IMF_Inflation_Rates.csv", deficit_data="IMF_Overall_Balance.csv", country_coding="Country_Coding.csv")
fig, ax = plot_dgs10_line(save_path='dgs10_plot.png', show=False)
scenario = wacc_calculator.calculate_wacc_scenarios()
low_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity="Low")
high_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity="High")
missing_values =  set(wacc_calculator.CRP['Country code'].unique().tolist()) ^ set(scenario['Country code'].unique().tolist())
name_value_dict = dict(zip(wacc_calculator.country_coding['Country code'], wacc_calculator.country_coding['Country Name']))
missing_names = [name_value_dict.get(code, code) for code in missing_values]

SSP = st.selectbox(
            "SSP", ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5"), 
            index=0, key="SSP", placeholder="Select SSP...")
country = st.selectbox(
            "Displayed Country", scenario["Country Name"].unique(), 
            index=0, placeholder="Select Country...", key="Country")
sensitivity = st.selectbox(
            "Scenario", ["Low", "Central", "High"], 
            index=0, placeholder="Select Low/Central/High Estimates...", key="Sensitivity")
tab1, tab11, tab2, tab3, tab4 = st.tabs(["Country-level CoE", "Country-level CoD","Regional Comparisons", "EMDEs", "Advanced Economies"])
if sensitivity == "Low":
    selected_scenario = low_scenario
elif sensitivity == "High":
    selected_scenario = high_scenario
else:
    selected_scenario = scenario

with tab1:
    # Select data based on input
    selected_data = selected_scenario.loc[(selected_scenario["Scenario"] == SSP) & (selected_scenario["Country Name"] == country)  & (selected_scenario["Technology"] == "Clean")]
    plot_comparison_chart_equity(selected_data[["Year", "Risk Free Rate", "Country Risk Premium", "Equity Risk Premium", "Technology Risk Premium"]])
with tab11:
    # Select data based on input
    selected_data = selected_scenario.loc[(selected_scenario["Scenario"] == SSP) & (selected_scenario["Country Name"] == country)  & (selected_scenario["Technology"] == "Clean")]
    plot_comparison_chart_debt(selected_data[["Year", "Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Technology Risk Premium"]])

with tab2:
    # Regional boxplots (easy to change year)
    years = sorted(selected_scenario['Year'].unique())
    default_index = years.index(2050) if 2050 in years else 0
    year_choice = st.selectbox('Year (for regional boxplots)', years, index=default_index)
    plot_region_boxplots_by_ssp_matplotlib(selected_scenario, year=year_choice, technology='Clean')

with tab3:
    # Comparison across EMDEs and Advanced Economies for both technologies
    scenario_comparison = selected_scenario.loc[selected_scenario["Country Name"].isin(["EMDEs", "Advanced Economies"])]
    plot_ssp_comparison(scenario_comparison[(scenario_comparison["Technology"] == "Clean")*(scenario_comparison["Country Name"] == "EMDEs")])
    # Pass full scenario_comparison (both Clean and Fossil) to create side-by-side subplots
    #plot_ssp_comparison_matplotlib(scenario_comparison)
    #plot_ssp_comparison_range_matplotlib(scenario, low_scenario, high_scenario, technology='Clean')

with tab4:
    plot_ssp_comparison(scenario_comparison[(scenario_comparison["Technology"] == "Clean")*(scenario_comparison["Country Name"] == "Advanced Economies")])






