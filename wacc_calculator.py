import pandas as pd
import numpy as np
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

class WaccCalculator:

    def __init__(self, GDP_data, SSP_data, CRP, CDS, tax_data, debt_data, inflation_data, deficit_data, country_coding):
        
        
        # Read in historical and future GDP data
        self.GDP_data = pd.read_csv("./DATA/" + GDP_data)
        self.convert_gdp_terms()
        self.SSP_data = pd.read_csv("./DATA/" + SSP_data, encoding="cp1252")

        # Read in historical CRP and CDS data
        self.CDS = pd.read_excel("./DATA/" + CDS, sheet_name="CDS")
        self.CRP = pd.read_excel("./DATA/" + CRP, sheet_name="CRP")

        # Read in tax data
        self.tax_data = pd.read_csv("./DATA/"+ tax_data)
        self.debt_data = pd.read_csv("./DATA/"+ debt_data)
        self.inflation_data = pd.read_csv("./DATA/"+ inflation_data)
        self.deficit_data = pd.read_csv("./DATA/"+ deficit_data)
        self.country_coding = pd.read_csv("./DATA/" + country_coding)


        # Set other assumptions
        self.category_margin = 3
        self.country_risk_gdp = self.evaluate_crp_gdp_v2()
        self.cds_gdp = self.evaluate_cds_gdp_v2()

        # Set technology maturity premiums
        self.technology_maturity = {
            "Mature": 0,
            "Commercial": 1.2,
            "Pre-Commercial": 2.4, 
            "Emerging": 3.6, 
            "FOAK": 4.8
        }


    def convert_gdp_terms(self):

        # Convert from 2015 to 2017 using IMF world inflation rates
        self.GDP_data[self.GDP_data.select_dtypes(include=['number']).columns] *= 1.027 * 1.033 

        # Change tracking
        self.GDP_data["Indicator Name"] = "GDP per capita (constant 2017 US$)"

    
    def evaluate_crp_gdp(self):

        def power_law(x, a, b):
            return a * x**b

        # Selected years
        selected_years = [str(year) for year in range(2015, 2025)]
        selected_years.append("Country code")
        
        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Merge CRP data with GDP data
        long_CRP = self.CRP.melt(id_vars=["Country code", "Country Risk Premium", "Country"], value_name="CRP", var_name="Year")
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code","Year"])

        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["GDP", "CRP"], axis="index")
        params, cov = curve_fit(power_law, merged_data["GDP"].values, merged_data["CRP"].values)
        X_0, exponent = params

        # Compute R2 Predictions
        y_pred = power_law( merged_data["GDP"].values, X_0, exponent)

        # R-squared calculation
        ss_res = np.sum((merged_data["CRP"].values - y_pred) ** 2)
        ss_tot = np.sum((merged_data["CRP"].values - np.mean(merged_data["CRP"].values)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        print(f"coefficient of determination: {r_squared}")
        print(f"Factor: {X_0}")
        print(f"Exponent: {exponent}")

        return X_0, exponent

    def plot_regression(self, model, log_gdp, inflation, deficit, debt, y_values, figname=None):

        log_gdp_grid = np.linspace(log_gdp.min(), log_gdp.max(), 100)

        infl_mean = inflation.mean()
        deficit_mean = deficit.mean()
        debt_mean = debt.mean()

        X_grid = np.column_stack([
        log_gdp_grid,
        np.full_like(log_gdp_grid, infl_mean),
        np.full_like(log_gdp_grid, deficit_mean),
        np.full_like(log_gdp_grid, debt_mean)
    ])


        # Predict and prepare x
        crp_pred_unclipped = model.predict(X_grid).ravel()
        x = np.exp(log_gdp_grid).ravel()

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
            x1, y1 = x[i-1], y[i-1]   # last positive point
            x2, y2 = x[i], y[i]       # first non-positive point

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

        #plt.figure(figsize=(8, 6))
        #plt.scatter(np.exp(log_gdp), y_values, alpha=0.6, label="Yearly data")
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
            return a * x**b

        # Selected years
        selected_years = [str(year) for year in range(2000, 2025)]
        selected_years.append("Country code")
        
        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Get inflation rates with years and country codes
        selected_inflation = self.inflation_data[selected_years].melt(id_vars="Country code", value_name='Inflation', var_name="Year")
        selected_inflation["Year"] = selected_inflation["Year"].astype(int)
        selected_inflation = selected_inflation[["Year" ,"Country code", "Inflation"]]

        # Get deficit rates with years and country codes
        selected_deficit = self.deficit_data[selected_years].melt(id_vars="Country code", value_name='Deficit', var_name="Year")
        selected_deficit["Year"] = selected_deficit["Year"].astype(int)
        selected_deficit = selected_deficit[["Year" ,"Country code", "Deficit"]]

        # Get government debt rates with years and country codes
        selected_debt = self.debt_data[selected_years].melt(id_vars="Country code", value_name='Debt', var_name="Year")
        selected_debt["Year"] = selected_debt["Year"].astype(int)
        selected_debt = selected_debt[["Year" ,"Country code", "Debt"]]



        # Merge CRP data with GDP data
        long_CRP = self.CRP.melt(id_vars=["Country code", "Country Risk Premium", "Country"], value_name="CRP", var_name="Year")
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code","Year"]).merge(
            selected_inflation, how="left", on=["Country code","Year"]).merge(
                selected_deficit, how="left", on=["Country code","Year"]).merge(
                    selected_debt, how="left", on=["Country code","Year"])


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
            return a * x**b

        # Selected years
        selected_years = [str(year) for year in range(2000, 2025)]
        selected_years.append("Country code")
        
        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Get inflation rates with years and country codes
        selected_inflation = self.inflation_data[selected_years].melt(id_vars="Country code", value_name='Inflation', var_name="Year")
        selected_inflation["Year"] = selected_inflation["Year"].astype(int)
        selected_inflation = selected_inflation[["Year" ,"Country code", "Inflation"]]

        # Get deficit rates with years and country codes
        selected_deficit = self.deficit_data[selected_years].melt(id_vars="Country code", value_name='Deficit', var_name="Year")
        selected_deficit["Year"] = selected_deficit["Year"].astype(int)
        selected_deficit = selected_deficit[["Year" ,"Country code", "Deficit"]]

        # Get government debt rates with years and country codes
        selected_debt = self.debt_data[selected_years].melt(id_vars="Country code", value_name='Debt', var_name="Year")
        selected_debt["Year"] = selected_debt["Year"].astype(int)
        selected_debt = selected_debt[["Year" ,"Country code", "Debt"]]



        # Merge CRP data with GDP data
        long_CRP = self.CDS.melt(id_vars=["Country code", "Country", "Rating-based Default Spread"], value_name="CDS", var_name="Year")
        long_CRP["Year"] = long_CRP["Year"].astype(int)
        merged_data = long_CRP.merge(selected_GDP, how="left", on=["Country code","Year"]).merge(
            selected_inflation, how="left", on=["Country code","Year"]).merge(
                selected_deficit, how="left", on=["Country code","Year"]).merge(
                    selected_debt, how="left", on=["Country code","Year"])


        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["GDP", "CDS", "Inflation", "Deficit", "Debt"], axis="index")
        X_values = np.column_stack([
            np.log(merged_data["GDP"]),
            merged_data["Inflation"],
            merged_data["Deficit"],
            merged_data["Debt"]
        ])

        y_values = merged_data["CDS"].values
        model = LinearRegression()
        model.fit(X_values, y_values)

        # Call plot function
        self.plot_regression(model, np.log(merged_data["GDP"]), merged_data["Inflation"], merged_data["Deficit"], merged_data["Debt"], y_values, figname="cds_regression.png")

        
        # Get values
        beta0 = model.intercept_
        gdp_coefficient, beta2, beta3, beta4 = model.coef_

        # Compute R2 Predictions
        print(f'coefficient of determination: {model.score(X_values, y_values)}')
        print(f"GDP coefficient: {gdp_coefficient}")

        return gdp_coefficient

    def evaluate_cds_gdp(self):

        def power_law(x, a, b):
            return a * x**b
        
        # Selected years
        selected_years = [str(year) for year in range(2015, 2025)]
        selected_years.append("Country code")
        
        # Get GDP values with years and country codes
        selected_GDP = self.GDP_data[selected_years].melt(id_vars="Country code", value_name='GDP', var_name="Year")
        selected_GDP["Year"] = selected_GDP["Year"].astype(int)

        # Merge CDS data with GDP data
        long_CDS = self.CDS.melt(id_vars=["Country code", "Country", "Rating-based Default Spread"], value_name="CDS", var_name="Year")
        long_CDS["Year"] = long_CDS["Year"].astype(int)
        merged_data = long_CDS.merge(selected_GDP, how="left", on=["Year", "Country code"])

        # Calculate relationship between crp and GDP
        merged_data = merged_data.dropna(subset=["GDP", "CDS"], axis="index")
        params, cov = curve_fit(power_law, merged_data["GDP"].values, merged_data["CDS"].values)
        X_0, exponent = params

        # Compute R2 Predictions
        y_pred = power_law( merged_data["GDP"].values, X_0, exponent)

        # R-squared calculation
        ss_res = np.sum((merged_data["CDS"].values - y_pred) ** 2)
        ss_tot = np.sum((merged_data["CDS"].values - np.mean(merged_data["CDS"].values)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        print(f"coefficient of determination: {r_squared}")
        print(f"Factor: {X_0}")
        print(f"Exponent: {exponent}")

        return X_0, exponent


    
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


    def calculate_aggregates(self, data):
        
        # Merge on country full names
        data = data.merge(self.country_coding.drop(columns=["Country Name"]), how="left", on="Country code")
        copied_data = data.copy()

        # Calculate median for each region and wb_income_group
        median_results_region = copied_data[["Region", "Scenario", "Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Equity Risk Premium", "Technology Risk Premium", "Overall Cost of Capital", "Year", "Technology", "Policy Coherence"]].groupby(['Scenario', 'Year', "Technology", 'Region', 'Policy Coherence']).mean().reset_index()
        median_results_region["Country Name"] = median_results_region["Region"] + " Mean"

        median_results_income = copied_data[["wb_income_group", "Scenario", "Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Equity Risk Premium", "Technology Risk Premium", "Overall Cost of Capital", "Year", "Technology", "Policy Coherence"]].groupby(['Scenario', 'Year', "Technology", 'Policy Coherence', 'wb_income_group']).mean().reset_index()
        median_results_income["Country Name"] = median_results_income["wb_income_group"] + " Mean"

        median_results_aggs = copied_data[["emde_advanced", "Scenario", "Risk Free Rate", "Country Risk Premium", "Lenders Margin", "Equity Risk Premium", "Technology Risk Premium", "Overall Cost of Capital", "Year", "Technology", "Policy Coherence"]].groupby(['Scenario', 'Year', "Technology", 'Policy Coherence', 'emde_advanced']).mean().reset_index()
        median_results_aggs["Country Name"] = median_results_aggs["emde_advanced"]
        aggregated_data = pd.concat([data, median_results_region, median_results_income, median_results_aggs], ignore_index=True)

        return aggregated_data

    def evaluate_risk_free(self, data, sensitivity=None):

        # Evaluate risk-free rate dependent on sensitivities
        if sensitivity == "High":
            rf_rate = 5.25
        elif sensitivity == "Low":
            rf_rate = 0.75
        else:
            rf_rate = 3
        
        # Set risk free rate
        data["Risk Free Rate"] = rf_rate

        return data
    
    def evaluate_instrument_parameters(self, data, sensitivity=None):

        # Set limits
        lm_low = 1.5
        lm_high = 2.5

        # Calculate based on country risk premium
        data["Lenders Margin"] = lm_low + (data["Country Risk Premium"] / data["Country Risk Premium"].max()) * (lm_high - lm_low)

        # Set limits for ERP
        ERP = self.CRP[self.CRP["Country code"] == "ERP"][range(2015, 2025)]
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
        policy_incoherence["Policy Coherence Premium"] = 2
        data["Policy Coherence Premium"] = 0

        # Set tracker
        policy_incoherence["Policy Coherence"] = "Weak"
        data["Policy Coherence"] = "Strong"

        # Append data
        merged_data = pd.concat([data, policy_incoherence], ignore_index=True)

        return merged_data



    def evaluate_total_costs(self, data, sensitivity=None):

        # Extract tax rate
        tax_rate = self.tax_data
        tax_rate.loc[tax_rate["2024"]=="NA", "2024"] = np.nanmean(tax_rate["2024"])

        # Merge tax rate onto main dataset
        data = data.merge(tax_rate[["Country code", "2024"]], how="left", on="Country code").rename(columns={"2024":"Tax Rate"})       

        # Calculate the debt share
        data["Debt Share"] = 80 - 40 * (data["Country Risk Premium"] - 
                                        data["Country Risk Premium"].min()) / (data["Country Risk Premium"].max() - 
                                                                               data["Country Risk Premium"].min())

        # Calculate the cost of debt
        data["Cost of Debt"] = data["Risk Free Rate"] + data["Country Default Spread"] + data["Lenders Margin"] + data["Technology Risk Premium"] + data["Policy Coherence Premium"]

        # Calculate the cost of equity
        mature_equity_tech_risk = 1.5
        data["Technology Risk Premium (E)"] = data["Technology Risk Premium"] + mature_equity_tech_risk
        data["Cost of Equity"] = data["Risk Free Rate"] + data["Country Risk Premium"] + data["Equity Risk Premium"] + data["Technology Risk Premium (E)"] + data["Policy Coherence Premium"]

        # Calculate the overall cost of capital
        data["Overall Cost of Capital"] = data["Debt Share"] / 100 * data["Cost of Debt"] * (1 - data["Tax Rate"]/100) + data["Cost of Equity"] * (1 - data["Debt Share"]/100)
        calculated_data = data.copy()

        # Merge on country full names
        calculated_data = calculated_data.merge(self.GDP_data[["Country Name", "Country code"]], how="left", on="Country code")

        return calculated_data


    def calculate_country_risk(self):

        # 1. Interpolate future GDP per capita ranges
        all_years = range(2025, 2101)
        future_GDP = self.SSP_data.copy().drop(["Region", "Variable", "Unit"], axis=1).set_index(["Model", "Scenario", "Country code"])
        future_GDP.columns = future_GDP.columns.astype(int)
        interpolated_GDP = future_GDP.reindex(columns=all_years).interpolate(axis=1).reset_index()
        collated_results = interpolated_GDP.loc[interpolated_GDP["Scenario"] != "Historical Reference", :].melt(id_vars=["Model", "Scenario", "Country code"], 
                                                                                                                var_name="Year", value_name="GDP per capita")

        # Extract 2025 values for GDP per capita and merge on
        gdp_results = collated_results.copy()
        gdp_2025 = gdp_results[(gdp_results["Year"]==2025) & (gdp_results["Scenario"]=="SSP1")].rename(columns={"GDP per capita":"GDP per capita 2025"})[["Country code", "GDP per capita 2025"]]
        collated_results = collated_results.merge(gdp_2025, how="left", on="Country code")

        # Extract 2025 values for CRP and CDS and merge on
        crp_2025 = self.CRP[["Country code", 2025]].rename(columns={2025:"Country Risk Premium"})
        cds_2025 = self.CDS[["Country code", 2025]].rename(columns={2025:"Country Default Spread"})
        collated_results = collated_results.merge(crp_2025, how="left", on="Country code")
        collated_results = collated_results.merge(cds_2025, how="left", on="Country code")

        # 2. Convert GDP per capita to country risk premium 
        collated_results["Country Risk Premium"] = collated_results["Country Risk Premium"] + self.country_risk_gdp * np.log(collated_results["GDP " \
        "per capita"] / collated_results["GDP per capita 2025"]) 

        # 3. Convert GDP per capita to country default spread
        collated_results["Country Default Spread"] = collated_results["Country Risk Premium"] + self.cds_gdp * np.log(collated_results["GDP " \
        "per capita"] / collated_results["GDP per capita 2025"]) 

        # 4. Drop intermediate columns
        collated_results = collated_results.drop(columns=["GDP per capita 2025"], axis=1)

        # Clip
        collated_results["Country Risk Premium"] = collated_results["Country Risk Premium"].clip(lower=0)
        collated_results["Country Default Spread"] = collated_results["Country Default Spread"].clip(lower=0)

        return collated_results

        