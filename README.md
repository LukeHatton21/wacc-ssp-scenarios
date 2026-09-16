# SSP-linked Country Risk and WACC Scenario Projections (2025–2100)

## 1. Overview

This repository contains input data and reproducible code used to generate country-level and aggregated projections of:

- Scenarios for the evolution of country risk premium and country default spreads
- Cost of debt and cost of equity estimates, for five technology maturity levels and two policy support schemes
- Weighted average cost of capital (WACC),

under the five Shared Socioeconomic Pathways (SSPs) scenarios from 2025 to 2100.

The dataset is produced using a Python pipeline (`ssp_scenarios.py`) that executes scenario calculation, visualization, 
and export steps sequentially, using the WaccCalculator object specified in (`wacc_calculator.py`).

---

## 2. Context and Purpose

Many energy system models and integrated assessment models use static discount rates for technology investments, despite 
empirical data demonstrating how it can vary substantially over time, across technologies and between countries. This 
dataset aims to address those challenges by developing long-run scenarios for the cost of capital linked t
to the Shared Socioeconomic Pathways (SSPs) used as a common basis for modelling that is collated by the IPCC. 
Outputs are provided at multiple aggregation levels to enable:

- cross-country comparison,
- regional and income-group distributional analysis,
- EMDE vs. Advanced Economy comparison,
- scenario sensitivity analysis (Low, Central, High),


---

## 3. Repository structure

Output folders produced by the pipeline:

- `./FINAL/`  
  CSV data products (long and wide format).
- `./PLOTS/`  
  Figures generated from the selected scenario run.

---

## 4. Code and computational workflow

Main executable script:

- `ssp_scenarios.py`

Primary entry point:

- `main_pipeline(...)`

Default behavior:

1. Computes central, low, and high WACC scenarios.
2. Produces benchmark and comparative figures:
   - Treasury benchmark time series,
   - EMDE/Advanced economy comparisons,
   - regional boxplots,
   - world heatmaps by SSP.
3. Exports harmonized CSV outputs in long and wide format.

Country-level interactive selection is intentionally removed from this pipeline version but is 
easily implementable under the existing formulas.

---

## 5. Input data dependencies

The pipeline expects the following source files (filenames as configured in code), which are all included in the repository:

- `GDP_Historical_PPP.csv`: Historical GDP per capita values, by country and year, in power purchasing parity terms
- `SSP_OECD_ENV_GDP_PC.csv`: GDP per capita scenarios under the SSPs
- `SSP_OECD_ENV_GDP.csv`: GDP scenarios under the SSPs
- `Collated_CRP_CDS.xlsx`: Historical country risk premium and country default spread data
- `CORPORATE_TAX_DATA.csv`: Data on corporate tax rates
- `IMF_Government_Debt.csv`: Government debt as a share of GDP, taken from the IMF
- `IMF_Inflation_Rates.csv`: Inflation rates at a national scale, taken from the IMF
- `IMF_Overall_Balance.csv`: Current account balances at a national scale, taken from the IMF
- `IMF_Gov_Primary_Balance.csv`
- `WBG_Debt_Servicing.csv`: Debt servicing as a percentage of GDP, taken from the World Bank
- `Country_Coding.csv`: Country mapping, from ISO 3 codes to country name, income status and regional grouping
- `DATA/DGS10.csv`: Historical US 10 year treasury yields, for benchmarking.

---

## 6. Output files and formats

### 6.1 Long-format scenario outputs

Generated in `./FINAL/`:

- `SSP_WACC_SCENARIOS_CENTRAL_<TECH>_LONG.csv`
- `SSP_WACC_SCENARIOS_LOW_<TECH>_LONG.csv`
- `SSP_WACC_SCENARIOS_HIGH_<TECH>_LONG.csv`

Where `<TECH>` is one of:
`MATURE`, `FOAK`, `EARLY COMMERCIAL`, `SCALING`, `COMMERCIAL`.

Each file contains annual estimates of the cost of capital by country, technology, scenario and policy 
maturity, with breakdowns of the contributing factors.

### 6.2 Wide-format scenario outputs

Generated in `./FINAL/`:

- `SSP_WACC_SCENARIOS_CENTRAL_WIDE.csv`
- `SSP_WACC_SCENARIOS_LOW_WIDE.csv`
- `SSP_WACC_SCENARIOS_HIGH_WIDE.csv`
- `SSP_WACC_SCENARIOS_ALL_WIDE.csv`

Each file contains annual estimates of the cost of capital by country, technology, scenario and policy 
maturity, pivoted into wide formats for easier comparison of the overall cost of capital over time.

### 6.3 Country-risk parameters

Generated in `./FINAL/`:

- `SSP_WACC_SCENARIOS_COUNTRY_RISKS_LONG.csv`
- `SSP_WACC_SCENARIOS_COUNTRY_RISKS_WIDE.csv`
- `SSP_WACC_SCENARIOS_COUNTRY_RISKS_LAGGEDCDS_WIDE.csv`

These focus on country risk premium and country default spread evolution, including a breakdown of how
the evolution changes when a different regression is used to relate country default spread
to GDP per capita (by including a lag of the country default spread).

### 6.4 Plot outputs

Generated in `./PLOTS/`:

- `dgs10_plot.png`
- `ssp_comparison_emde_ae.html`
- `ssp_comparison_matplotlib.png`
- `ssp_comparison_range_matplotlib.png`
- `boxplots_region_aggregates_<TECH>_<YEAR>_<POLICY>.png`
- `wacc_world_heatmap_<YEAR>-<TECH>-<POLICY>-<SENSITIVITY>.png`

---

## 7. Core output variables (selected)

Representative fields include:

- `Country Name`
- `Country code`
- `Region`
- `WBG Income Group (2025)`
- `Scenario`
- `Year`
- `Technology`
- `Policy Maturity`
- `GDP per capita (USD2017/pc, PPP)`
- `Total GDP (USD billion PPP, 2017)`
- `Risk Free Rate`
- `Country Risk Premium`
- `Country Default Spread`
- `Country Risk Premium (Lagged)` (where relevant). Not included in the main estimates.
- `Country Default Spread (Lagged)` (where relevant). Not included in the main estimates.
- `Cost of Debt`
- `Cost of Equity`
- `Overall Cost of Capital`


## 9. Geospatial mapping notes

World heatmaps use Natural Earth administrative boundaries via GeoPandas-compatible loading.  
To ensure compatibility with GeoPandas >= 1.0, the script loads Natural Earth from URL (or user-specified local path) rather than deprecated built-in datasets.

## 10 Execution

Run:

```bash
python ssp_scenarios.py