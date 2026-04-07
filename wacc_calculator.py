import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from linearmodels.panel import PanelOLS
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr


class WaccCalculator:

    def __init__(self, GDP_data, SSP_data, CRP, CDS, tax_data, debt_data, inflation_data, deficit_data, revenue_data,
                 servicing_data, country_coding, SSP_GDP):

        # Read in historical and future GDP data
        self.GDP_data = pd.read_csv("./DATA/" + GDP_data)
        self.convert_gdp_terms()
        self.SSP_data = pd.read_csv("./DATA/" + SSP_data, encoding="cp1252")
        self.SSP_data = self.SSP_data.loc[self.SSP_data["Unit"]=="USD_2017/yr"]
        self.GDP_SSP_data = pd.read_csv("./DATA/"+SSP_GDP, encoding="cp1252")
        self.GDP_SSP_data = self.GDP_SSP_data.loc[self.GDP_SSP_data["Unit"] == "billion USD_2017/yr"]

        # Read in historical CRP and CDS data
        self.CDS = pd.read_excel("./DATA/" + CDS, sheet_name="CDS")
        self.CRP = pd.read_excel("./DATA/" + CRP, sheet_name="CRP")

        # Read in tax data
        self.tax_data = pd.read_csv("./DATA/" + tax_data)
        self.debt_data = pd.read_csv("./DATA/" + debt_data)
        self.inflation_data = pd.read_csv("./DATA/" + inflation_data)
        self.deficit_data = pd.read_csv("./DATA/" + deficit_data)
        self.revenue_data = pd.read_csv("./DATA/" + revenue_data)
        self.debt_servicing_data = pd.read_csv("./DATA/" + servicing_data)
        self.country_coding = pd.read_csv("./DATA/" + country_coding)
        self.gdp_growth = pd.read_csv("./DATA/IMF_GDP_Growth.csv")

        self.gdp = pd.read_csv("./DATA/IMF_GDP.csv")
        self.population = pd.read_csv("./DATA/IMF_Population.csv")

        # Set other assumptions
        self.category_margin = 3

        # Set technology maturity premiums
        self.technology_maturity = {
            "Mature": 0,
            "Commercial": 1.2,
            "Scaling": 2.4,
            "Early Commercial": 3.6,
            "FOAK": 5.1
        }

    def convert_gdp_terms(self):

        # Convert from 2015 to 2017 using IMF world inflation rates
        self.GDP_data[self.GDP_data.select_dtypes(include=['number']).columns] *= 1.027 * 1.033

        # Change tracking
        self.GDP_data["Indicator Name"] = "GDP per capita (constant 2017 US$)"

    def convert_gdp_terms_ppp(self):

        # Convert from 2015 to 2017 using IMF world inflation rates
        self.GDP_data[self.GDP_data.select_dtypes(include=['number']).columns] /= (1.019 * 1.024 * 1.028 * 1.029)

        # Change tracking
        self.GDP_data["Indicator Name"] = "GDP per capita PPP (constant 2017 US$)"

    def evaluate_crp_gdp(self):

        def power_law(x, a, b):
            return a * x ** b

        # Selected years
        selected_years = [str(year) for year in range(2015, 2025)]
        selected_years.append("Country code")

        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Merge CRP data with GDP data
        long_CRP = self.CRP.melt(id_vars=["Country code", "Country Risk Premium", "Country"], value_name="CRP",
                                 var_name="Year")
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code", "Year"])

        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["GDP", "CRP"], axis="index")
        params, cov = curve_fit(power_law, merged_data["GDP"].values, merged_data["CRP"].values)
        X_0, exponent = params

        # Compute R2 Predictions
        y_pred = power_law(merged_data["GDP"].values, X_0, exponent)

        # R-squared calculation
        ss_res = np.sum((merged_data["CRP"].values - y_pred) ** 2)
        ss_tot = np.sum((merged_data["CRP"].values - np.mean(merged_data["CRP"].values)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        print(f"coefficient of determination: {r_squared}")
        print(f"Factor: {X_0}")
        print(f"Exponent: {exponent}")

        return X_0, exponent

    def plot_log_regression(self, model, log_gdp, inflation, deficit, debt, y_values, gdp_threshold=60000,
                            figname=None):
        """
        Plots CDS vs log(GDP), with a constant CDS line after a GDP threshold.
        x-axis is log(GDP), but tick labels show actual GDP for readability.
        """
        # Create grid over log(GDP)
        log_gdp_grid = np.linspace(log_gdp.min(), log_gdp.max(), 200)

        # Use mean values for other predictors
        infl_mean = inflation.mean()
        deficit_mean = deficit.mean()
        debt_mean = debt.mean()

        X_grid = np.column_stack([
            log_gdp_grid,
            np.full_like(log_gdp_grid, infl_mean),
            np.full_like(log_gdp_grid, deficit_mean),
            np.full_like(log_gdp_grid, debt_mean)
        ])

        # Predict CDS
        cds_pred = model.predict(X_grid).ravel()

        # Convert threshold GDP to log(GDP)
        log_gdp_threshold = np.log(gdp_threshold)

        # Split grid into below and above threshold
        below_mask = log_gdp_grid <= log_gdp_threshold
        above_mask = log_gdp_grid > log_gdp_threshold

        # Below threshold
        x_below = log_gdp_grid[below_mask]
        y_below = cds_pred[below_mask]

        # Above threshold (flat CDS)
        x_above = log_gdp_grid[above_mask]
        if len(x_above) > 0:
            y_above = np.full_like(x_above, cds_pred[below_mask][-1])

        # Plot scatter of raw data
        plt.figure(figsize=(8, 6))
        plt.scatter(log_gdp, y_values, alpha=0.6, label="Yearly data")

        # Plot fitted regression line
        plt.plot(x_below, y_below, color="red", linewidth=2, label="Fitted regression")

        # Plot constant line above threshold
        if len(x_above) > 0:
            plt.plot(x_above, y_above, color="red", linewidth=2, linestyle="--",
                     label=f"Constant CDS above GDP={gdp_threshold:,}")

        # Axes labels
        plt.xlabel("GDP per capita (constant 2017 US$, thousand p.c.)")
        plt.ylabel("Country Default Spread (%)")

        # X-axis ticks in GDP units
        gdp_ticks = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000]
        log_ticks = np.log(gdp_ticks)
        plt.xticks(log_ticks, [f"{tick / 1000:.0f}" for tick in gdp_ticks])

        # Y-axis formatting
        plt.yticks(np.arange(0, 26, 5))
        plt.legend()
        plt.tight_layout()

        if figname:
            plt.savefig(figname, dpi=300)
        plt.show()

    def plot_log_regression_v2(self, result, log_gdp, inflation, deficit, debt, servicing, revenue, y_values,
                               gdp_threshold=24500, figname=None):
        """
        Plots CDS vs log(GDP), with a constant CDS line after a GDP threshold.
        x-axis is log(GDP), but tick labels show actual GDP for readability.
        Uses a PanelOLS result object from linearmodels.
        """
        # Create grid over log(GDP)
        log_gdp_grid = np.linspace(log_gdp.min(), log_gdp.max(), 200)
        median_log_gdp = np.nanmedian(log_gdp)
        tolerance = (log_gdp.max() - log_gdp.min()) * 0.05  # 5% window around median
        near_median_mask = np.abs(log_gdp - median_log_gdp) < tolerance
        anchor_y = np.nanmedian(y_values[near_median_mask]) if near_median_mask.any() else np.nanmedian(y_values)

        # Predict CDS over the grid, holding other predictors at their median
        y_pred = (
                result.params["Log_GDP"] * log_gdp_grid +
                result.params["Inflation"] * np.nanmedian(inflation) +
                result.params["Deficit"] * np.nanmedian(deficit) +
                result.params["Debt"] * np.nanmedian(debt) +
                result.params["Servicing"] * np.nanmedian(servicing) +
                result.params["Revenue"] * np.nanmedian(revenue)
        )

        y_pred_at_median = (
                result.params["Log_GDP"] * median_log_gdp +
                result.params["Inflation"] * np.nanmedian(inflation) +
                result.params["Deficit"] * np.nanmedian(deficit) +
                result.params["Debt"] * np.nanmedian(debt) +
                result.params["Servicing"] * np.nanmedian(servicing) +
                result.params["Revenue"] * np.nanmedian(revenue))

        # Anchor to fitted values mean (accounts for absorbed fixed effects)
        shift = anchor_y - y_pred_at_median
        y_pred = y_pred + shift

        # Find threshold where y_pred first goes below 0
        below_zero = np.where(y_pred < 0)[0]
        if len(below_zero) > 0:
            gdp_threshold = np.exp(log_gdp_grid[below_zero[0]])  # convert back from log
        else:
            gdp_threshold = np.exp(log_gdp_grid[-1])  # fallback: end of grid

        log_gdp_threshold = np.log(gdp_threshold)

        # Split grid into below and above threshold
        below_mask = log_gdp_grid <= log_gdp_threshold
        above_mask = log_gdp_grid > log_gdp_threshold

        # Below threshold
        x_below = log_gdp_grid[below_mask]
        y_below = y_pred[below_mask]

        # Above threshold (flat CDS — capped at the last predicted value before threshold)
        x_above = log_gdp_grid[above_mask]
        if len(x_above) > 0:
            y_above = np.full_like(x_above, y_pred[below_mask][-1])

        # Plot scatter of raw data
        plt.figure(figsize=(8, 6))
        plt.scatter(log_gdp, y_values, alpha=0.6, label="Yearly data")

        # Plot fitted regression line
        plt.plot(x_below, y_below, color="red", linewidth=2, label="Fitted regression")

        # Plot constant line above threshold
        if len(x_above) > 0:
            plt.plot(x_above, y_above, color="red", linewidth=2, linestyle="--",
                     label=f"Limit of modelled GDP - CRP relationship)")

        # Axes labels
        plt.xlabel("GDP per capita (constant 2017 US$, thousand p.c.)")
        plt.ylabel("Country Default Spread (%)")

        # X-axis ticks in GDP units
        gdp_ticks = [1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000]
        log_ticks = np.log(gdp_ticks)
        plt.xticks(log_ticks, [f"{tick / 1000:.0f}" for tick in gdp_ticks])

        # Y-axis formatting
        plt.yticks(np.arange(0, 26, 5))
        plt.legend()
        plt.tight_layout()

        if figname:
            plt.savefig(figname, dpi=300)
        plt.show()

    def make_regression_table_v2(
            self,
            results: list,
            model_names: list,
            panel_data: pd.DataFrame,
            dep_var: str,
            clustered: bool = True,
    ) -> pd.DataFrame:
        """
        Build a publication-style regression table from a list of PanelOLS results.

        Args:
            results:     list of fitted PanelOLS result objects
            model_names: list of column headers e.g. ["(1)", "(2)", "(3)", "(4)"]
            panel_data:  the panel dataframe used for estimation
            dep_var:     name of the dependent variable column e.g. "CDS"
            clustered:   whether standard errors are clustered (for table footer note)
        """
        import numpy as np

        # ── 1. Coefficient + SE rows ──────────────────────────────────────────────
        all_vars = []
        for res in results:
            all_vars.extend(res.params.index.tolist())
        all_vars = list(dict.fromkeys(all_vars))  # deduplicate, preserve order

        rows = {}
        for var in all_vars:
            coef_row, se_row = [], []
            for res in results:
                if var in res.params:
                    coef = res.params[var]
                    se = res.std_errors[var]
                    pval = res.pvalues[var]

                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                    elif pval < 0.1:
                        stars = "."
                    else:
                        stars = ""

                    coef_row.append(f"{coef:.4f}{stars}")
                    se_row.append(f"({se:.4f})")
                else:
                    coef_row.append("")
                    se_row.append("")

            rows[var] = coef_row
            rows[f"se_{var}"] = se_row

        # ── 2. Fixed-effects indicators ───────────────────────────────────────────
        rows["Country FE"] = ["No"] + ["Yes"] * (len(results) - 1)
        rows["Year FE"] = ["No"] + ["Yes"] * (len(results) - 1)

        # ── 3. Clustered SE indicator ─────────────────────────────────────────────
        rows["Clustered SE"] = ["Yes" if clustered else "No"] * len(results)

        # ── 4. Helper: overall R² ─────────────────────────────────────────────────
        def overall_r2(res):
            fitted = res.fitted_values
            resid = res.resids
            y = panel_data[dep_var].dropna().loc[fitted.index]
            ss_res = (resid ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            return 1 - ss_res / ss_tot

        # ── 5. Helper: log-likelihood ─────────────────────────────────────────────
        def log_likelihood(res):
            """Gaussian log-likelihood (consistent with linearmodels internals)."""
            n = res.nobs
            resid = res.resids
            sigma2 = (resid ** 2).sum() / n
            return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

        # ── 6. Helper: AIC / BIC ─────────────────────────────────────────────────
        def aic_bic(res):
            ll = log_likelihood(res)
            k = len(res.params)  # free slope parameters
            n = res.nobs
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)
            return aic, bic

        # ── 7. Fit statistics ─────────────────────────────────────────────────────
        rows["Observations"] = [f"{int(res.nobs)}" for res in results]
        rows["R²"] = [f"{overall_r2(res):.5f}" for res in results]
        rows["Within R²"] = [f"{res.rsquared_within:.5f}" for res in results]

        # RMSE
        rows["RMSE"] = [
            f"{np.sqrt((res.resids ** 2).mean()):.5f}" for res in results
        ]

        # F-statistic (joint significance of all regressors)
        rows["F-statistic"] = [
            f"{res.f_statistic.stat:.4f}" if hasattr(res, "f_statistic") else ""
            for res in results
        ]
        rows["F p-value"] = [
            f"{res.f_statistic.pval:.4f}" if hasattr(res, "f_statistic") else ""
            for res in results
        ]

        # AIC and BIC
        aic_vals, bic_vals = zip(*[aic_bic(res) for res in results])
        rows["AIC"] = [f"{v:.2f}" for v in aic_vals]
        rows["BIC"] = [f"{v:.2f}" for v in bic_vals]

        # Number of unique entities and time periods
        try:
            entity_idx = panel_data.index.get_level_values(0)
            time_idx = panel_data.index.get_level_values(1)
            rows["N entities"] = [f"{entity_idx.nunique()}"] * len(results)
            rows["N periods"] = [f"{time_idx.nunique()}"] * len(results)
        except Exception:
            pass  # silently skip if index is not a MultiIndex

        # ── 8. Assemble ───────────────────────────────────────────────────────────
        table = pd.DataFrame(rows, index=model_names).T
        table.index.name = "Variable"
        return table

    def make_regression_table(self, results: list, model_names: list, panel_data: pd.DataFrame,
                              dep_var: str) -> pd.DataFrame:
        """
        Build a publication-style regression table from a list of PanelOLS results.

        Args:
            results:     list of fitted PanelOLS result objects
            model_names: list of column headers e.g. ["(1)", "(2)", "(3)", "(4)"]
            panel_data:  the panel dataframe used for estimation
            dep_var:     name of the dependent variable column e.g. "CDS"
        """
        all_vars = []
        for res in results:
            all_vars.extend(res.params.index.tolist())
        all_vars = list(dict.fromkeys(all_vars))  # deduplicate, preserve order

        rows = {}
        for var in all_vars:
            coef_row = []
            se_row = []
            for res in results:
                if var in res.params:
                    coef = res.params[var]
                    se = res.std_errors[var]
                    pval = res.pvalues[var]

                    # Significance stars
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                    elif pval < 0.1:
                        stars = "."
                    else:
                        stars = ""

                    coef_row.append(f"{coef:.4f}{stars}")
                    se_row.append(f"({se:.4f})")
                else:
                    coef_row.append("")
                    se_row.append("")

            rows[var] = coef_row
            rows[f"se_{var}"] = se_row

        # Fixed effects rows
        rows["Country FE"] = ["No"] + ["Yes"] * (len(results) - 1)
        rows["Year FE"] = ["No"] + ["Yes"] * (len(results) - 1)

        # Fit statistics
        def fixest_r2(res):
            fitted = res.fitted_values
            resid = res.resids
            y = panel_data[dep_var].dropna()
            y = y.loc[fitted.index]
            ss_res = (resid ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            return 1 - ss_res / ss_tot

        rows["Observations"] = [f"{int(res.nobs)}" for res in results]
        rows["R²"] = [f"{fixest_r2(res):.5f}" for res in results]
        rows["Within R²"] = [f"{res.rsquared_within:.5f}" for res in results]

        table = pd.DataFrame(rows, index=model_names).T
        table.index.name = "Variable"
        return table

    def plot_regression(self, result, log_gdp, inflation, deficit, debt, servicing, revenue, y_values, figname=None):

        log_gdp_grid = np.linspace(log_gdp.min(), log_gdp.max(), 100)
        infl_mean = inflation.mean()
        deficit_mean = deficit.mean()
        debt_mean = debt.mean()
        servicing_mean = servicing.mean()
        revenue_mean = revenue.mean()

        X_grid = np.column_stack([
            log_gdp_grid,
            np.full_like(log_gdp_grid, infl_mean),
            np.full_like(log_gdp_grid, deficit_mean),
            np.full_like(log_gdp_grid, debt_mean),
            np.full_like(log_gdp_grid, servicing_mean),
            np.full_like(log_gdp_grid, revenue_mean)
        ])

        # Predict and prepare x
        log_gdp_grid_flat = log_gdp_grid.ravel()
        crp_pred_unclipped = (
                result.params["Log_GDP"] * log_gdp_grid_flat +
                result.params["Inflation"] * np.mean(inflation) +  # hold others at median
                result.params["Deficit"] * np.mean(deficit) +
                result.params["Debt"] * np.median(debt) +
                result.params["Debt"] * np.median(servicing) +
                result.params["Debt"] * np.medan(revenue)
        )

        x = np.exp(log_gdp_grid_flat).ravel()

        # Sort by x (important for a clean line)
        order = np.argsort(x)
        x = x[order]
        y = crp_pred_unclipped[order]

        plt.figure(figsize=(8, 6))
        plt.scatter(np.exp(log_gdp), y_values, alpha=0.6, label="Yearly data")

        # Find first crossing where y <= 0
        cross_idxs = np.where(y <= 0)[0]

        if len(cross_idxs) == 0:
            # No crossing: plot full unclipped line
            plt.plot(x, y, color="red", linewidth=2, label="Unclipped regression")
        elif cross_idxs[0] == 0:
            # Starts at or below 0: plot fully clipped line
            plt.plot(x, np.zeros_like(x), color="black", linewidth=2, linestyle="--",
                     label="Clipped at 0")
        else:
            i = cross_idxs[0]  # first index where y <= 0
            x1, y1 = x[i - 1], y[i - 1]  # last positive point
            x2, y2 = x[i], y[i]  # first non-positive point

            # Linear interpolation for exact crossing x
            x_cross = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)

            # Segment 1: unclipped until crossing (ends at y=0 for continuity)
            x_seg1 = np.concatenate([x[:i], [x_cross]])
            y_seg1 = np.concatenate([y[:i], [0.0]])
            plt.plot(x_seg1, y_seg1, color="red", linewidth=2, label="Fitted line from \nregression model")

            # Segment 2: clipped flat line at 0 from crossing onward
            x_seg2 = np.concatenate([[x_cross], x[i:]])
            y_seg2 = np.zeros_like(x_seg2)
            plt.plot(x_seg2, y_seg2, color="red", linewidth=2, linestyle="--",
                     label="Limit of modelled\nGDP - CRP relationship")

        # plt.figure(figsize=(8, 6))
        # plt.scatter(np.exp(log_gdp), y_values, alpha=0.6, label="Yearly data")
        plt.xlabel("GDP per capita (constant 2017 US$)")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(tkr.StrMethodFormatter('{x:,.0f}'))
        plt.xticks([0, 20000, 40000, 60000, 80000, 100000])
        plt.yticks([0, 5, 10, 15, 20, 25])
        plt.ylabel("Country Default Spread (%) ")
        plt.legend()
        plt.tight_layout()
        if figname is not None:
            plt.savefig(figname)

    def evaluate_crp_gdp_v2(self):

        def power_law(x, a, b):
            return a * x ** b

        # Selected years
        selected_years = [str(year) for year in range(2000, 2025)]
        selected_years.append("Country code")

        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Get inflation rates with years and country codes
        selected_inflation = self.inflation_data[selected_years].melt(id_vars="Country code", value_name='Inflation',
                                                                      var_name="Year")
        selected_inflation["Year"] = selected_inflation["Year"].astype(int)
        selected_inflation = selected_inflation[["Year", "Country code", "Inflation"]]

        # Get deficit rates with years and country codes
        selected_deficit = self.deficit_data[selected_years].melt(id_vars="Country code", value_name='Deficit',
                                                                  var_name="Year")
        selected_deficit["Year"] = selected_deficit["Year"].astype(int)
        selected_deficit = selected_deficit[["Year", "Country code", "Deficit"]]

        # Get government debt rates with years and country codes
        selected_debt = self.debt_data[selected_years].melt(id_vars="Country code", value_name='Debt', var_name="Year")
        selected_debt["Year"] = selected_debt["Year"].astype(int)
        selected_debt = selected_debt[["Year", "Country code", "Debt"]]

        # Merge CRP data with GDP data
        long_CRP = self.CRP.melt(id_vars=["Country code", "Country Risk Premium", "Country"], value_name="CRP",
                                 var_name="Year")
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code", "Year"]).merge(
            selected_inflation, how="left", on=["Country code", "Year"]).merge(
            selected_deficit, how="left", on=["Country code", "Year"]).merge(
            selected_debt, how="left", on=["Country code", "Year"])

        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["GDP", "CRP", "Inflation", "Deficit", "Debt"], axis="index")
        X_values = np.column_stack([
            np.log(merged_data["GDP"]),
            merged_data["Inflation"],
            merged_data["Deficit"],
            merged_data["Debt"]
        ])

        y_values = merged_data["CRP"].values
        model = LinearRegression()
        model.fit(X_values, y_values)

        # Get values
        beta0 = model.intercept_
        gdp_coefficient, beta2, beta3, beta4 = model.coef_

        # Compute R2 Predictions
        print(f'coefficient of determination: {model.score(X_values, y_values)}')
        print(f"GDP coefficient: {gdp_coefficient}")

        return gdp_coefficient

    def evaluate_cds_gdp_v2(self):

        def power_law(x, a, b):
            return a * x ** b

        # Selected years
        selected_years = [str(year) for year in range(2000, 2025)]
        selected_years.append("Country code")

        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int) - 1

        # Get inflation rates with years and country codes
        selected_inflation = self.inflation_data[selected_years].melt(id_vars="Country code", value_name='Inflation',
                                                                      var_name="Year")
        selected_inflation["Year"] = selected_inflation["Year"].astype(int)
        selected_inflation = selected_inflation[["Year", "Country code", "Inflation"]]

        # Get deficit rates with years and country codes
        selected_deficit = self.deficit_data[selected_years].melt(id_vars="Country code", value_name='Deficit',
                                                                  var_name="Year")
        selected_deficit["Year"] = selected_deficit["Year"].astype(int)
        selected_deficit = selected_deficit[["Year", "Country code", "Deficit"]]

        # Get government debt rates with years and country codes
        selected_debt = self.debt_data[selected_years].melt(id_vars="Country code", value_name='Debt', var_name="Year")
        selected_debt["Year"] = selected_debt["Year"].astype(int)
        selected_debt = selected_debt[["Year", "Country code", "Debt"]]

        # Get government debt servicing rates with years and country codes
        selected_servicing = self.debt_servicing_data[selected_years].melt(id_vars="Country code",
                                                                           value_name='Servicing', var_name="Year")
        selected_servicing["Year"] = selected_servicing["Year"].astype(int)
        selected_servicing = selected_servicing[["Year", "Country code", "Servicing"]]

        # Get government revenue rates (primary balance) with years and country codes
        selected_revenue = self.revenue_data[selected_years].melt(id_vars="Country code", value_name='Revenue',
                                                                  var_name="Year")
        selected_revenue["Year"] = selected_revenue["Year"].astype(int)
        selected_revenue = selected_revenue[["Year", "Country code", "Revenue"]]

        # Get historical GDP total rates with years and country codes
        selected_GDP_tot = self.gdp[selected_years].melt(id_vars="Country code", value_name='GDP_Total',
                                                         var_name="Year")
        selected_GDP_tot["Year"] = selected_GDP_tot["Year"].astype(int)
        selected_GDP_tot = selected_GDP_tot[["Year", "Country code", "GDP_Total"]]

        # Get historical GDP total rates with years and country codes
        selected_GDP_growth = self.gdp_growth[selected_years].melt(id_vars="Country code", value_name='GDP_Growth',
                                                         var_name="Year")
        selected_GDP_growth["Year"] = selected_GDP_growth["Year"].astype(int)
        selected_GDP_growth = selected_GDP_growth[["Year", "Country code", "GDP_Growth"]]

        # Get government revenue rates with years and country codes
        selected_pop = self.population[selected_years].melt(id_vars="Country code", value_name='Population',
                                                            var_name="Year")
        selected_pop["Year"] = selected_pop["Year"].astype(int)
        selected_pop = selected_pop[["Year", "Country code", "Population"]]

        # Merge CRP data with GDP data
        CDS_data = self.CDS.copy()

        # Apply mask for repeated values
        # selected_years_orig = [int(y) for y in selected_years_orig]
        # mask = CDS_data[selected_years_orig].eq(CDS_data[selected_years_orig].shift(axis=1))
        # CDS_data[selected_years_orig] = CDS_data[selected_years_orig].where(~mask)
        long_CRP = CDS_data.melt(id_vars=["Country code", "Country", "Rating-based Default Spread"], value_name="CDS",
                                 var_name="Year")

        long_CRP_lagged = long_CRP.copy().rename(columns={"CDS": "Lagged_CDS"})
        long_CRP_lagged["Year"] = long_CRP_lagged["Year"].astype(int) - 1

        # Account for year delay in CDS
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code", "Year"]).merge(
            selected_inflation, how="left", on=["Country code", "Year"]).merge(
            selected_deficit, how="left", on=["Country code", "Year"]).merge(
            selected_debt, how="left", on=["Country code", "Year"]).merge(
            selected_servicing, how="left", on=["Country code", "Year"]).merge(
            selected_revenue, how="left", on=["Country code", "Year"]).merge(
            long_CRP_lagged[["Country code", "Year", "Lagged_CDS"]], how="left", on=["Country code", "Year"]).merge(
            selected_GDP_tot, how="left", on=["Country code", "Year"]).merge(
            selected_pop, how="left", on=["Country code", "Year"]).merge(
            selected_GDP_growth, how="left", on=["Country code", "Year"])

        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["CDS", "Inflation", "Deficit", "Debt", "Revenue", "GDP_Total", "GDP_Growth"],
                                         axis="index", how="any")
        merged_data["weight"] = 1
        merged_data["Log_GDP"] = np.log(merged_data["GDP"])

        panel_data = merged_data.set_index(["Country code", "Year"])
        weights_df = panel_data["GDP"] / panel_data["GDP"].groupby(level=1).transform("max")
        # weights_df = panel_data[["weight"]]
        fe_model = PanelOLS.from_formula(
            "CDS ~ Log_GDP + Inflation + Deficit + Debt + Revenue + EntityEffects + TimeEffects",
            data=panel_data, weights=weights_df)
        result = fe_model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        print(result.summary)

        # Calculate with previous year
        fe_model_lagged = PanelOLS.from_formula(
            "CDS ~ Log_GDP + Inflation + Deficit + Debt + Revenue + EntityEffects + TimeEffects",
            data=panel_data, weights=weights_df)
        result_lagged = fe_model_lagged.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
        print(result_lagged.summary)

        # Coefficients
        gdp_beta = result_lagged.params["Log_GDP"]
        gdp_coefficient = gdp_beta

        # Call plot function
        # merged_data = merged_data.loc[merged_data["Year"] == 2022]
        y_values = merged_data["CDS"].values
        # self.plot_log_regression_v2(result, np.log(merged_data["GDP"]), merged_data["Inflation"], merged_data["Deficit"], merged_data["Debt"],
        # merged_data["Servicing"], merged_data["Revenue"], y_values, figname="cds_regression.png")



        # --- Run models ---
        panel_data = merged_data.set_index(["Country code", "Year"])

        fit0 = PanelOLS.from_formula("CDS ~ Log_GDP                                                           ",
                                     data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                                              cluster_time=True)
        fit1 = PanelOLS.from_formula("CDS ~ Log_GDP                              + EntityEffects + TimeEffects",
                                     data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                                              cluster_time=True)

        fit2 = PanelOLS.from_formula("CDS ~ Log_GDP + Inflation                  + EntityEffects + TimeEffects",
                                     data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                                              cluster_time=True)
        fit3 = PanelOLS.from_formula("CDS ~ Log_GDP + Inflation + Deficit        + EntityEffects + TimeEffects",
                                     data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                                              cluster_time=True)
        fit4 = PanelOLS.from_formula("CDS ~ Log_GDP + Inflation + Deficit + Debt + EntityEffects + TimeEffects",
                                     data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                                              cluster_time=True)
        fit6 = PanelOLS.from_formula(
            "CDS ~ Log_GDP + Inflation + Deficit + Debt + Revenue + EntityEffects + TimeEffects", data=panel_data,
            weights=weights_df).fit(cov_type="clustered", cluster_entity=True)
        fitmain = PanelOLS.from_formula(
            "CDS ~ Log_GDP + Inflation + Deficit + Debt + Revenue + EntityEffects + TimeEffects",
            data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True,
                                                     cluster_time=True)
        fit7 = PanelOLS.from_formula(
            "CDS ~ Log_GDP + Inflation + Deficit + Debt + Revenue + Lagged_CDS + EntityEffects + TimeEffects",
            data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True)
        fit8 = PanelOLS.from_formula(
            "CDS ~ Log_GDP  + Lagged_CDS +EntityEffects + TimeEffects",
            data=panel_data, weights=weights_df).fit(cov_type="clustered", cluster_entity=True)

        # --- Build and display table ---
        table = self.make_regression_table_v2(
            results=[fit0, fit1, fit2, fit3, fit4, fit6, fitmain, fit7, fit8],
            model_names=["(0)\nRegression", "(1)\nWith FE", "(3)", "(4)", "(5)", "(6)", "Main", "(7)", "(8)"],
            panel_data=panel_data,
            dep_var="CDS"
        )

        print(table.to_string())
        print("\nClustered (Country) standard errors in parentheses")
        print("Signif. Codes: ***: 0.001, **: 0.01, *: 0.05, .: 0.1")

        return gdp_coefficient

    def calculate_wacc_scenarios(self, sensitivity=None):

        # 1. Calculate country risk premiums and country default spreads
        calculated_data = self.calculate_country_risk()

        # 2. Calculate risk free rate
        calculated_data = self.evaluate_risk_free(data=calculated_data, sensitivity=sensitivity)

        # 3. Calculate lenders margin and equity risk premium
        calculated_data = self.evaluate_instrument_parameters(data=calculated_data, sensitivity=sensitivity)

        # 4. Calculate technology category premium
        calculated_data = self.evaluate_tech_risk(data=calculated_data, sensitivity=sensitivity)

        # 5. Calculate policy coherence premium
        calculated_data = self.evaluate_policy_coherence(data=calculated_data, sensitivity=sensitivity)

        # 6. Sum all to give the total
        calculated_data = self.evaluate_total_costs(data=calculated_data, sensitivity=sensitivity)

        # 6. Merge additional data and calculate medians
        calculated_data = self.calculate_aggregates(calculated_data)

        return calculated_data

    def calculate_weighted_mean(self, data, weight_col, cols, group_cols):

        df = data.copy()

        # create weighted columns
        for c in cols:
            df[f"{c}_w"] = df[c] * df[weight_col]

        # aggregation dictionary
        agg_dict = {f"{col}_w": "sum" for col in cols}
        agg_dict[weight_col] = "sum"

        tmp = df.groupby(group_cols).agg(agg_dict)

        # compute weighted means
        for col in cols:
            tmp[col] = tmp[f"{col}_w"] / tmp[weight_col]

        median_results_region = tmp.reset_index()[group_cols + cols]

        return median_results_region

    def calculate_aggregates(self, data):

        # Merge on country full names
        data = data.merge(self.country_coding.drop(columns=["Country Name"]), how="left", on="Country code")
        copied_data = data.copy()


        # Calculate median for each region and wb_income_group
        weight_col = "Total GDP (SSP)"
        cols = ["Risk Free Rate", "Country Risk Premium", "Lenders Margin",
                "Equity Risk Premium", "Technology Risk Premium",
                "Overall Cost of Capital"]
        group_cols = ['Scenario', 'Year', 'Technology', 'Region', 'Policy Maturity']
        median_results_region = self.calculate_weighted_mean(copied_data, weight_col, cols, group_cols)
        median_results_region["Country Name"] = median_results_region["Region"] + " Mean"

        cols = ["Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Equity Risk Premium",
                "Technology Risk Premium", "Overall Cost of Capital"]
        group_cols = ['Scenario', 'Year', "Technology", 'Policy Maturity', 'wb_income_group']
        median_results_income = self.calculate_weighted_mean(copied_data, weight_col, cols, group_cols)
        median_results_income["Country Name"] = median_results_income["wb_income_group"] + " Mean"

        cols = ["Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Equity Risk Premium",
                "Technology Risk Premium", "Overall Cost of Capital", ]
        group_cols = ['Scenario', 'Year', "Technology", 'Policy Maturity', 'emde_advanced']
        median_results_aggs = self.calculate_weighted_mean(copied_data, weight_col, cols, group_cols)
        median_results_aggs["Country Name"] = median_results_aggs["emde_advanced"]
        aggregated_data = pd.concat([data, median_results_region, median_results_income, median_results_aggs],
                                    ignore_index=True)

        return aggregated_data

    def evaluate_risk_free(self, data, sensitivity=None):

        # Evaluate risk-free rate dependent on sensitivities
        if sensitivity == "High":
            rf_rate = 4.8
        elif sensitivity == "Low":
            rf_rate = 0.6
        else:
            rf_rate = 2.5

        # Set risk free rate
        data["Risk Free Rate"] = rf_rate

        return data

    def evaluate_instrument_parameters(self, data, sensitivity=None):

        # Set limits
        lm_low = 1.5
        lm_high = 2.5

        # Calculate based on country risk premium
        data["Lenders Margin"] = lm_low + (data["Country Risk Premium"] / data["Country Risk Premium"].max()) * (
                    lm_high - lm_low)

        # Set limits for ERP
        ERP = self.CRP[self.CRP["Country code"] == "ERP"][range(2015, 2026)]
        if sensitivity == "High":
            erp_uniform = ERP.values.max()
        elif sensitivity == "Low":
            erp_uniform = ERP.values.min()
        else:
            erp_uniform = ERP.values.mean()

        # Set ERP rate
        data["Equity Risk Premium"] = erp_uniform

        return data

    def evaluate_tech_risk(self, data, sensitivity=None):

        # List to collect per-technology dataframes
        df_list = []

        for tech, premium in self.technology_maturity.items():
            # Duplicate the original data for this technology
            tech_df = data.copy()

            # Set the Technology column
            tech_df["Technology"] = tech

            # Set the Technology Risk Premium for all rows/years
            tech_df["Technology Risk Premium"] = premium

            df_list.append(tech_df)

        # Concatenate all technology-specific dataframes
        new_data = pd.concat(df_list, ignore_index=True)

        return new_data

    def evaluate_policy_coherence(self, data, sensitivity=None):

        # Copy data
        policy_incoherence = data.copy()

        # Set policy incoherence premium
        policy_incoherence["Policy Maturity Premium"] = 2
        data["Policy Maturity Premium"] = 0

        # Set tracker
        policy_incoherence["Policy Maturity"] = "Weak"
        data["Policy Maturity"] = "Strong"

        # Append data
        merged_data = pd.concat([data, policy_incoherence], ignore_index=True)

        return merged_data

    def evaluate_total_costs(self, data, sensitivity=None):

        # Extract tax rate
        tax_rate = self.tax_data
        tax_rate.loc[tax_rate["2024"] == "NA", "2024"] = np.nanmean(tax_rate["2024"])

        # Merge tax rate onto main dataset
        data = data.merge(tax_rate[["Country code", "2024"]], how="left", on="Country code").rename(
            columns={"2024": "Tax Rate"})

        # Calculate the debt share
        data["Debt Share"] = 80 - 40 * (data["Country Risk Premium"] -
                                        data["Country Risk Premium"].min()) / (data["Country Risk Premium"].max() -
                                                                               data["Country Risk Premium"].min())

        # Calculate the cost of debt
        data["Cost of Debt"] = data["Risk Free Rate"] + data["Country Default Spread"] + data["Lenders Margin"] + data[
            "Technology Risk Premium"] + data["Policy Maturity Premium"]

        # Calculate the cost of equity
        mature_equity_tech_risk = 1.5
        data["Technology Risk Premium (E)"] = data["Technology Risk Premium"] + mature_equity_tech_risk
        data["Cost of Equity"] = data["Risk Free Rate"] + data["Country Risk Premium"] + data["Equity Risk Premium"] + \
                                 data["Technology Risk Premium (E)"] + data["Policy Maturity Premium"]

        # Calculate the overall cost of capital
        data["Overall Cost of Capital"] = data["Debt Share"] / 100 * data["Cost of Debt"] * (
                    1 - data["Tax Rate"] / 100) + data["Cost of Equity"] * (1 - data["Debt Share"] / 100)
        calculated_data = data.copy()

        # Merge on country full names
        calculated_data = calculated_data.merge(self.country_coding[["Country Name", "Country code"]], how="left",
                                                on="Country code")

        return calculated_data

    def calculate_country_risk(self):

        # 1. Interpolate future GDP per capita ranges
        all_years = range(2025, 2101)
        future_GDP = self.SSP_data.copy().drop(["Region", "Variable", "Unit", "version", "Type"], axis=1).set_index(
            ["Model", "Scenario", "Country code"])
        future_GDP.columns = future_GDP.columns.astype(int)
        interpolated_GDP = future_GDP.reindex(columns=all_years).interpolate(axis=1).reset_index()
        collated_results = interpolated_GDP.loc[interpolated_GDP["Scenario"] != "Historical Reference", :].melt(
            id_vars=["Model", "Scenario", "Country code"],
            var_name="Year", value_name="GDP per capita")


        # Do the same for total GDP
        future_GDP_total = self.GDP_SSP_data.copy().drop(["Region", "Variable", "Unit", "version", "Type", "Model"],
                                                   axis=1).set_index(
            ["Scenario", "Country code"])
        future_GDP_total.columns = future_GDP_total.columns.astype(int)
        interpolated_GDP_total = future_GDP_total.reindex(columns=all_years).interpolate(axis=1).reset_index()
        melted_GDP_total = interpolated_GDP_total.melt(id_vars=["Scenario", "Country code"], var_name="Year",
                                           value_name="Total GDP (SSP)")

        # Merge GDP total onto GDP per capita
        collated_results = collated_results.merge(melted_GDP_total, how="left", on=["Year", "Scenario", "Country code"])

        # Extract 2025 values for GDP per capita and merge on
        gdp_results = collated_results.copy()
        gdp_2025 = gdp_results[(gdp_results["Year"] == 2025) & (gdp_results["Scenario"] == "SSP1")].rename(
            columns={"GDP per capita": "GDP per capita 2025"})[["Country code", "GDP per capita 2025"]]
        collated_results = collated_results.merge(gdp_2025, how="left", on="Country code")

        # Extract 2025 values for CRP and CDS and merge on
        crp_2025 = self.CRP[["Country code", 2025]].rename(columns={2025: "Country Risk Premium"})
        cds_2025 = self.CDS[["Country code", 2025]].rename(columns={2025: "Country Default Spread"})
        collated_results = collated_results.merge(crp_2025, how="left", on="Country code")
        collated_results = collated_results.merge(cds_2025, how="left", on="Country code")
        collated_results["Country Risk Premium (2025)"] = collated_results[
            "Country Risk Premium"]
        collated_results["Country Default Spread (2025)"] = collated_results[
            "Country Default Spread"]

        # Get values for gdp to cds relationship through regression
        self.cds_gdp = self.evaluate_cds_gdp_v2()
        crp_coefficient = 1.35

        # 2. Convert GDP per capita to country risk premium
        collated_results["Country Risk Premium"] = collated_results[
                                                       "Country Risk Premium"] + self.cds_gdp * crp_coefficient * np.log(
            np.maximum(collated_results["GDP " \
                             "per capita"] / collated_results["GDP per capita 2025"], 1))

        # 3. Convert GDP per capita to country default spread
        collated_results["Country Default Spread"] = collated_results["Country Default Spread"] + self.cds_gdp * np.log(
            np.maximum(collated_results["GDP " \
                             "per capita"] / collated_results["GDP per capita 2025"],1))

        # Clip results for zero
        collated_results["Country Risk Premium"] = collated_results["Country Risk Premium"].clip(lower=0)
        collated_results["Country Default Spread"] = collated_results["Country Default Spread"].clip(lower=0)

        # 4. Drop intermediate columns
        collated_results = collated_results.drop(columns=["GDP per capita 2025"], axis=1)


        return collated_results

