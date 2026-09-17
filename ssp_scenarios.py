# Updated version: country-level selections removed from pipeline execution.
# This script runs regional boxplots, EMDE/Advanced Economy comparisons,
# world heatmaps, and long/wide CSV exports sequentially.

import os
from pathlib import Path as _Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as _pd
import geopandas as gpd
import matplotlib.dates as mdates

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from wacc_calculator import WaccCalculator


# =========================
# Global configuration
# =========================
SSP_COLOR_MAP = {
    'SSP1': '#2ca02c',  # green
    'SSP2': '#ff7f0e',  # orange
    'SSP3': '#1f77b4',  # blue
    'SSP4': '#9467bd',  # purple
    'SSP5': '#d62728',  # red
}


# =========================
# Helpers
# =========================
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_altair_chart(chart, save_path=None):
    if save_path:
        chart.save(save_path)


def safe_show(show=False):
    if show:
        plt.show()


# =========================
# Plotting functions
# =========================
def plot_ssp_comparison(df, save_path=None):
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

    save_altair_chart(chart, save_path)
    return chart


def plot_ssp_comparison_matplotlib(df, figsize=(16, 4), save_path='ssp_comparison_matplotlib.png', show=False):
    technologies = ['FOAK', 'Early Commercial', 'Scaling', 'Commercial', 'Mature']
    countries = ['EMDEs', 'Advanced Economies']
    policy = ['Strong']

    fig, axes = plt.subplots(1, 5, figsize=figsize, sharey=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    scenarios = sorted(df['Scenario'].unique())
    linestyles = ['-', '--', '-.', ':']
    tab_colors = plt.get_cmap('tab10')(range(10))
    country_styles = {country: linestyles[i % len(linestyles)] for i, country in enumerate(countries)}

    df = df.loc[df['Policy Maturity'].isin(policy)]
    for ax, tech in zip(axes, technologies):
        for i, scenario in enumerate(scenarios):
            color = SSP_COLOR_MAP.get(scenario, tab_colors[i % len(tab_colors)])
            for country in countries:
                data = df[(df['Scenario'] == scenario) & (df['Country Name'] == country) & (df['Technology'] == tech)]
                if data.empty:
                    continue
                ax.plot(data['Year'], data['Overall Cost of Capital'],
                        linestyle=country_styles[country], color=color, linewidth=1.5)
        ax.set_title(f'{tech}')
        ax.set_xlabel('Year')
        ax.set_xticks([2025, 2050, 2075, 2100])

    axes[0].set_ylabel('Mean Cost of Capital (%, nominal)')

    scenario_handles = [Line2D([0], [0], color=SSP_COLOR_MAP.get(scenarios[i], tab_colors[i % len(tab_colors)]), lw=2)
                        for i in range(len(scenarios))]
    country_handles = [Line2D([0], [0], color='k', lw=1.5, linestyle=country_styles[c]) for c in countries]

    handles = scenario_handles + country_handles
    labels = scenarios + countries
    fig.legend(handles, labels, loc='lower center', ncol=max(3, len(labels)), bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.subplots_adjust(bottom=0.2, wspace=0.15)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    safe_show(show)
    plt.close(fig)
    return fig, axes


def plot_ssp_comparison_range_matplotlib(scenario, low_scenario, high_scenario, technology='Mature', include_both=False,
                                         figsize=(12, 10), save_path='ssp_comparison_range_matplotlib.png', show=False):
    countries = ['EMDEs', 'Advanced Economies']
    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[2, :])]

    tab_colors = plt.get_cmap('tab10')(range(10))
    country_styles = {'EMDEs': '--', 'Advanced Economies': '-'}

    for idx, ssp in enumerate(ssp_list):
        ax = axes[idx]
        ax.set_title(ssp)
        ax.set_xlabel('Year')
        color = SSP_COLOR_MAP.get(ssp, tab_colors[idx % len(tab_colors)])

        techs = ['Mature', 'Pre-Commercial'] if include_both else [technology]

        for tech in techs:
            for country in countries:
                central = scenario[(scenario['Scenario'] == ssp) & (scenario['Country Name'] == country) & (scenario.get('Technology') == tech)].copy()
                low = low_scenario[(low_scenario['Scenario'] == ssp) & (low_scenario['Country Name'] == country) & (low_scenario.get('Technology') == tech)].copy()
                high = high_scenario[(high_scenario['Scenario'] == ssp) & (high_scenario['Country Name'] == country) & (high_scenario.get('Technology') == tech)].copy()

                if central.empty:
                    continue

                central['Year'] = _pd.to_numeric(central['Year'], errors='coerce')
                central = central.dropna(subset=['Year'])
                central['Year'] = central['Year'].astype(int)
                years = list(central['Year'].unique())

                low['Year'] = _pd.to_numeric(low['Year'], errors='coerce')
                high['Year'] = _pd.to_numeric(high['Year'], errors='coerce')

                low_series = low.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)
                high_series = high.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)
                central_series = central.groupby('Year')['Overall Cost of Capital'].mean().reindex(years)

                low_series = _pd.to_numeric(low_series, errors='coerce').astype(float)
                high_series = _pd.to_numeric(high_series, errors='coerce').astype(float)
                central_series = _pd.to_numeric(central_series, errors='coerce').astype(float)

                low_series = low_series.interpolate(limit_direction='both').fillna(central_series)
                high_series = high_series.interpolate(limit_direction='both').fillna(central_series)

                years_arr = np.array(years, dtype=float)
                low_arr = low_series.to_numpy(dtype=float)
                high_arr = high_series.to_numpy(dtype=float)
                central_arr = central_series.to_numpy(dtype=float)

                ax.fill_between(years_arr, low_arr, high_arr, color=color, alpha=0.12, linewidth=0)
                if tech.lower() == 'pre-commercial':
                    ax.plot(years_arr, central_arr, color=color, lw=1.5, linestyle=country_styles[country],
                            marker='o', markersize=4, markerfacecolor=color, markeredgecolor='k')
                else:
                    ax.plot(years_arr, central_arr, color=color, lw=2, linestyle=country_styles[country])

        ax.grid(alpha=0.35, linestyle='--')
        ax.set_ylabel('Cost of Capital (%, nominal)')
        ax.set_yticks([0, 5, 10, 15])

    country_handles = [Line2D([0], [0], color='k', lw=1.5, linestyle=country_styles[c], label=c) for c in countries]
    fossil_handle = Line2D([0], [0], color='k', lw=1.5, marker='o', label='Fossil', markerfacecolor='k', markersize=5)
    envelope_patch = Patch(facecolor='gray', alpha=0.12, label='Low / High range')

    handles = country_handles + ([fossil_handle] if include_both else []) + [envelope_patch]
    labels = [h.get_label() for h in handles]

    fig.subplots_adjust(bottom=0.25, hspace=0.5, wspace=0.25)
    axes[-1].legend(handles, labels, loc='right', ncol=1, bbox_to_anchor=(1.55, 0.5), frameon=False)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    safe_show(show)
    plt.close(fig)
    return fig, axes


def plot_dgs10_line(filepath=None, start_date=None, end_date=None, rolling=None, figsize=(10, 6), ax=None, show=False,
                    save_path=None):
    path = _Path(filepath) if filepath is not None else _Path(__file__).resolve().parent / 'DATA' / 'DGS10.csv'
    df = _pd.read_csv(path, parse_dates=['observation_date'])
    df = df.sort_values('observation_date').dropna(subset=['DGS10'])

    if start_date is not None:
        df = df[df['observation_date'] >= _pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['observation_date'] <= _pd.to_datetime(end_date)]

    x = df['observation_date']
    y = _pd.to_numeric(df['DGS10'], errors='coerce')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(x, y, color='C0', linewidth=2)
    if rolling and isinstance(rolling, int) and rolling > 1:
        ax.plot(x, y.rolling(window=rolling).mean(), label=f'{rolling}-day MA', color='green', linewidth=2)

    ax.set_xlabel('Date', fontsize=20)
    ax.set_ylabel('10-Year Treasury Yield (%, nominal)', fontsize=20)
    ax.grid(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=0)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.text(0.02, 0.94, 'a', transform=ax.transAxes, fontsize=20, fontweight='bold')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    safe_show(show)
    plt.close(fig)
    return fig, ax


def plot_region_boxplots_by_ssp_matplotlib(df, policy, year=2050, regions=None, technology='Mature',
                                           save_path=None, show=False):
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

    if 'Region' not in df.columns:
        return None, None

    plot_df = df.copy()
    plot_df['Country Name'] = plot_df['Country Name'].astype(str).str.strip()
    plot_df['Region'] = plot_df['Region'].astype(str).str.strip()

    try:
        plot_df['Year'] = plot_df['Year'].astype(int)
    except Exception:
        pass

    plot_df = plot_df[plot_df['Year'] == int(year)]
    if technology is not None:
        plot_df = plot_df[plot_df['Technology'] == technology]

    plot_df = plot_df[~plot_df['Country Name'].str.contains('Mean', na=False)]
    plot_df = plot_df.loc[(plot_df['Technology'] == technology) & (plot_df['Policy Maturity'] == policy)]
    plot_df = plot_df[plot_df['Region'].isin(regions)]

    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    plot_df = plot_df[plot_df['Scenario'].isin(ssp_list)]

    if plot_df.empty:
        return None, None

    income_categories = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
    combined_df = plot_df.copy()
    if 'wb_income_group' in plot_df.columns:
        income_df = plot_df[plot_df['wb_income_group'].notna()].copy()
        income_df['Region'] = income_df['wb_income_group']
        combined_df = _pd.concat([income_df, combined_df], ignore_index=True)

    final_regions = [ic for ic in income_categories if ic in combined_df['Region'].unique()]
    final_regions += [r for r in regions if r not in final_regions]

    n_regions = len(final_regions)
    y = np.arange(n_regions)
    n_ssp = len(ssp_list)
    width = 0.13
    colors = [SSP_COLOR_MAP.get(ssp, plt.get_cmap('Set2')(i / max(1, n_ssp - 1))) for i, ssp in enumerate(ssp_list)]

    fig, ax = plt.subplots(figsize=(10, max(6, n_regions * 0.6)))

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
            medianprops={'color': 'black', 'linewidth': 1.2},
            showfliers=False,
        )
        for patch in bp['boxes']:
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(final_regions)
    ax.set_xlabel(f'Overall Cost of Capital (%, {year}, {technology}, {policy} Policy)')

    legend_patches = [Patch(facecolor=colors[i], label=ssp_list[i]) for i in range(n_ssp)]
    ax.legend(handles=legend_patches, title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    safe_show(show)
    plt.close(fig)
    return fig, ax

def load_world_geometries(world_path=None):
    """
    Load Natural Earth country polygons robustly across GeoPandas versions.

    Priority:
    1) user-provided local file path
    2) Natural Earth 110m admin-0 countries ZIP URL
    """
    if world_path:
        return gpd.read_file(world_path)

    ne_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    return gpd.read_file(ne_url)

def plot_wacc_world_heatmap(selected_scenario, year_choice, technology='Mature', figsize=(20, 12),
                            save_path='wacc_world_heatmap', show=False, vmin=None, vmax=None,
                            policy=None, sensitivity='Central'):
    world = load_world_geometries()
    ssp_list = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']

    plot_df = selected_scenario[
        (selected_scenario['Year'] == year_choice) &
        (selected_scenario['Technology'] == technology)
    ].copy()

    if plot_df.empty:
        return None, None
    if 'Country code' not in plot_df.columns:
        return None, None

    code_column = 'Country code'
    plot_df_agg = plot_df.groupby(['Scenario', code_column])['Overall Cost of Capital'].mean().reset_index()

    fig, axes = plt.subplots(5, 1, figsize=figsize)

    if vmin is None:
        vmin = plot_df_agg['Overall Cost of Capital'].min()
    if vmax is None:
        vmax = plot_df_agg['Overall Cost of Capital'].max()

    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, ssp in enumerate(ssp_list):
        ax = axes[idx]
        ssp_data = plot_df_agg[plot_df_agg['Scenario'] == ssp].copy()
        world_plot = world.merge(ssp_data, left_on='ADM0_A3', right_on=code_column, how='left')

        world.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.5)
        world_with_data = world_plot[world_plot['Overall Cost of Capital'].notna()]
        world_no_data = world_plot[world_plot['Overall Cost of Capital'].isna()]

        world_with_data.plot(ax=ax, column='Overall Cost of Capital', cmap=cmap, norm=norm,
                             edgecolor='#333333', linewidth=0.3, legend=False)
        world_no_data.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3, hatch='////')

        ax.set_title(f'{ssp} - {technology}, {policy} Policy ({year_choice})', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-65, 90])
        ax.set_xlim([-180, 180])

    fig.suptitle(f'World Map Heatmap: ({technology}, {year_choice}, {sensitivity})', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    first_ax = axes[0]
    ax_pos = first_ax.get_position()
    cbar_ax = fig.add_axes([ax_pos.x0, 0.94, ax_pos.width, 0.015])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', extend='max')
    cbar.set_label('Overall Cost of Capital (%)', fontsize=10)

    out_path = f'{save_path}_{year_choice}-{technology}-{policy}-{sensitivity}.png'
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    safe_show(show)
    plt.close(fig)
    return fig, axes


# =========================
# Export logic
# =========================
def export_long_outputs(scenario, low_scenario, high_scenario, final_dir='./FINAL'):
    rename = {
        'wb_income_group': 'WBG Income Group (2025)',
        'Total GDP (SSP)': 'Total GDP (USD billion PPP, 2017)',
        'GDP per capita': 'GDP per capita (USD2017/pc, PPP)'
    }

    selected_columns = [
        'Country Name', 'Country code', 'Region', 'WBG Income Group (2025)',
        'Scenario', 'Year', 'Technology', 'Policy Maturity',
        'GDP per capita (USD2017/pc, PPP)', 'Total GDP (USD billion PPP, 2017)',
        'Risk Free Rate', 'Country Risk Premium', 'Country Default Spread',
        'Equity Risk Premium', 'Technology Risk Premium', 'Lenders Margin', "Tax Rate", "Debt Share", 'Cost of Debt', 'Cost of Equity', 'Overall Cost of Capital'
    ]

    for tech in ['Mature', 'FOAK', 'Early Commercial', 'Scaling', 'Commercial']:
        central = scenario.loc[(scenario['Technology'] == tech)].rename(columns=rename)
        low = low_scenario.loc[(low_scenario['Technology'] == tech)].rename(columns=rename)
        high = high_scenario.loc[(high_scenario['Technology'] == tech)].rename(columns=rename)

        central = central[selected_columns].apply(lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))
        low = low[selected_columns].apply(lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))
        high = high[selected_columns].apply(lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))

        central.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_CENTRAL_{tech.upper()}_LONG.csv', index=False)
        low.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_LOW_{tech.upper()}_LONG.csv', index=False)
        high.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_HIGH_{tech.upper()}_LONG.csv', index=False)

def export_country_risk_scenarios(scenario, final_dir='./FINAL'):
    wide_selected = [
        'Country Name', 'Country code', 'Region', 'wb_income_group',
        'Scenario', 'Technology', 'Policy Maturity', 'Overall Cost of Capital', 'Year'
    ]

    rename = {
        'wb_income_group': 'WBG Income Group (2025)',
        'Total GDP (SSP)': 'Total GDP (USD billion PPP, 2017)',
        'GDP per capita': 'GDP per capita (USD2017/pc, PPP)'
    }

    selected_columns = [
        'Country Name', 'Country code', 'Region', 'WBG Income Group (2025)',
        'Scenario', 'Year',
        'GDP per capita (USD2017/pc, PPP)', 'Total GDP (USD billion PPP, 2017)', 'Country Risk Premium', 'Country Default Spread', 'Country Risk Premium (Lagged)', 'Country Default Spread (Lagged)',
    ]
    remove = ['Country Risk Premium', 'Country Default Spread', 'Country Risk Premium (Lagged)', 'Country Default Spread (Lagged)']
    selected_columns_wide = [x for x in selected_columns if x not in remove]
    def _to_wide(df, value="Country Default Spread"):
        out = df[['Country Name', 'Country code', 'Region', 'WBG Income Group (2025)', 'Scenario', 'Year',value]].pivot_table(
            index=['Country Name', 'Country code', 'Region', 'WBG Income Group (2025)','Scenario'],
            columns=['Year'], values=value, aggfunc='mean'
        )
        out = out.apply(lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))
        return out

    # Rename and remove duplicates by selecting only for one technology policy combination
    country_risk_scenario = scenario.loc[(scenario['Technology'] == "Mature") & (scenario['Policy Maturity'] == "Strong")].rename(columns=rename).drop(columns=["Technology", "Policy Maturity"])
    country_risk_scenario = country_risk_scenario[selected_columns].apply(
        lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))

    # Produce wide version of the country risk scenario (main)
    country_risk_scenario_wide = _to_wide(country_risk_scenario)
    country_risk_wide_lagged = _to_wide(country_risk_scenario, value="Country Default Spread (Lagged)")

    # Save to the output folder
    country_risk_scenario.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_COUNTRY_RISKS_LONG.csv', index=False)
    country_risk_scenario_wide.reset_index().to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_COUNTRY_DEFAULT_SPREAD_WIDE.csv', index=False)
    country_risk_wide_lagged.reset_index().to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_COUNTRY__DEFAULT_SPREAD_LAGGEDCDS_WIDE.csv', index=False)




def export_wide_outputs(scenario, low_scenario, high_scenario, final_dir='./FINAL'):
    rename = {
        'wb_income_group': 'WBG Income Group (2025)',
        'Total GDP (SSP)': 'Total GDP (USD billion PPP, 2017)',
        'GDP per capita': 'GDP per capita (USD2017/pc, PPP)'
    }

    wide_selected = [
        'Country Name', 'Country code', 'Region', 'wb_income_group',
        'Scenario', 'Technology', 'Policy Maturity', 'Overall Cost of Capital', 'Year'
    ]

    scenario = scenario.copy()
    low_scenario = low_scenario.copy()
    high_scenario = high_scenario.copy()

    scenario.fillna(value={'wb_income_group': 'N/A', 'Country code': 'N/A', 'Region': 'N/A'}, inplace=True)
    low_scenario.fillna(value={'wb_income_group': 'N/A', 'Country code': 'N/A', 'Region': 'N/A'}, inplace=True)
    high_scenario.fillna(value={'wb_income_group': 'N/A', 'Country code': 'N/A', 'Region': 'N/A'}, inplace=True)

    def _to_wide(df):
        out = df[wide_selected].rename(columns=rename, errors='ignore').pivot_table(
            index=['Country Name', 'Country code', 'Region', 'WBG Income Group (2025)', 'Scenario', 'Technology', 'Policy Maturity'],
            columns=['Year'], values=['Overall Cost of Capital'], aggfunc='mean'
        )
        out = out.apply(lambda col: col.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x))
        return out

    central_wide = _to_wide(scenario)
    low_wide = _to_wide(low_scenario)
    high_wide = _to_wide(high_scenario)

    central_wide.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_CENTRAL_WIDE.csv')
    low_wide.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_LOW_WIDE.csv')
    high_wide.to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_HIGH_WIDE.csv')
    _pd.concat([central_wide, low_wide, high_wide]).to_csv(f'{final_dir}/SSP_WACC_SCENARIOS_ALL_WIDE.csv')


# =========================
# Main pipeline
# =========================
def main_pipeline(
    technology='Mature',
    policy='Strong',
    sensitivity='Central',
    years_for_plots=(2030, 2050, 2100),
    save_plots=True,
    show_plots=False
):
    ensure_dirs('./PLOTS', './FINAL')

    wacc_calculator = WaccCalculator(
        GDP_data='GDP_Historical_PPP.csv',
        SSP_data='SSP_OECD_ENV_GDP_PC.csv',
        CRP='Collated_CRP_CDS.xlsx',
        CDS='Collated_CRP_CDS.xlsx',
        tax_data='CORPORATE_TAX_DATA.csv',
        debt_data='IMF_Government_Debt.csv',
        inflation_data='IMF_Inflation_Rates.csv',
        deficit_data='IMF_Overall_Balance.csv',
        revenue_data='IMF_Gov_Primary_Balance.csv',
        servicing_data='WBG_Debt_Servicing.csv',
        country_coding='Country_Coding.csv',
        SSP_GDP='SSP_OECD_ENV_GDP.csv'
    )

    # 1) Risk-free benchmark figure
    plot_dgs10_line(save_path='./PLOTS/dgs10_plot.png' if save_plots else None, show=show_plots)

    # 2) Scenario calculations
    scenario = wacc_calculator.calculate_wacc_scenarios()
    low_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity='Low')
    high_scenario = wacc_calculator.calculate_wacc_scenarios(sensitivity='High')

    selected_scenario = {'Low': low_scenario, 'Central': scenario, 'High': high_scenario}[sensitivity]

    # 3) EMDE / Advanced Economy comparisons (no individual-country plotting)
    scenario_comparison = selected_scenario.loc[selected_scenario['Country Name'].isin(['EMDEs', 'Advanced Economies'])]
    if save_plots is True:
        plot_ssp_comparison(
            scenario_comparison[
                (scenario_comparison['Policy Maturity'] == policy) &
                (scenario_comparison['Technology'] == technology)
            ],
            save_path='./PLOTS/ssp_comparison_emde_ae.html' if save_plots else None
        )

        plot_ssp_comparison_matplotlib(
            scenario_comparison,
            save_path='./PLOTS/ssp_comparison_matplotlib.png' if save_plots else None,
            show=show_plots
        )

        plot_ssp_comparison_range_matplotlib(
            scenario, low_scenario, high_scenario,
            technology=technology,
            include_both=False,
            save_path='./PLOTS/ssp_comparison_range_matplotlib.png' if save_plots else None,
            show=show_plots
        )

        # 4) Regional boxplots + world maps in loop
        for year in years_for_plots:
            plot_region_boxplots_by_ssp_matplotlib(
                selected_scenario,
                policy=policy,
                year=year,
                technology=technology,
                save_path=f'./PLOTS/boxplots_region_aggregates_{technology}_{year}_{policy}.png' if save_plots else None,
                show=show_plots
            )

            plot_wacc_world_heatmap(
                selected_scenario,
                year,
                technology=technology,
                figsize=(20, 12),
                save_path='./PLOTS/wacc_world_heatmap',
                show=show_plots,
                vmin=4,
                vmax=16,
                policy=policy,
                sensitivity=sensitivity
            )

    # 5) Exports (long + wide)
    export_country_risk_scenarios(scenario, final_dir='./FINAL')
    export_long_outputs(scenario, low_scenario, high_scenario, final_dir='./FINAL')
    export_wide_outputs(scenario, low_scenario, high_scenario, final_dir='./FINAL')

    return {
        'scenario': scenario,
        'low_scenario': low_scenario,
        'high_scenario': high_scenario,
        'selected_scenario': selected_scenario
    }


if __name__ == '__main__':
    main_pipeline(
        technology='Mature',
        policy='Strong',
        sensitivity='Central',
        years_for_plots=(2030, 2050, 2100),
        save_plots=True,
        show_plots=False
    )
