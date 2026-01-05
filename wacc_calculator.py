import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

class WaccCalculator:

    def __init__(self, GDP_data, SSP_data, CRP, CDS):
        
        
        # Read in historical and future GDP data
        self.GDP_data = pd.read_csv("./DATA/" + GDP_data)
        self.convert_gdp_terms()
        self.SSP_data = pd.read_csv("./DATA/" + SSP_data, encoding="cp1252")

        # Read in historical CRP and CDS data
        self.CDS = pd.read_excel("./DATA/" + CDS, sheet_name="CDS")
        self.CRP = pd.read_excel("./DATA/" + CRP, sheet_name="CRP")

        # Set other assumptions
        self.category_margin = 3
        self.country_risk_gdp, self.country_risk = self.evaluate_crp_gdp()
        self.cds_gdp, self.cds = self.evaluate_cds_gdp()


    def convert_gdp_terms(self):

        # Convert from 2015 to 2017
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


    
    def calculate_wacc_scenarios(self):

        # 1. Calculate country risk premiums and country default spreads
        calculated_data = self.calculate_country_risk()

        # 2. Calculate risk free rate

        # 3. Calculate lenders margin and equity risk premium

        # 4. Calculate technology category premium

        # 5. Sum all to give the total

    def calculate_country_risk(self):

        # 1. Interpolate future GDP per capita ranges
        all_years = range(2025, 2101)
        future_GDP = self.SSP_data.copy().drop(["Region", "Variable", "Unit"], axis=1).set_index(["Model", "Scenario", "Country code"])
        future_GDP.columns = future_GDP.columns.astype(int)
        interpolated_GDP = future_GDP.reindex(columns=all_years).interpolate(axis=1).reset_index()
        collated_results = interpolated_GDP.loc[interpolated_GDP["Scenario"] != "Historical Reference", :].melt(id_vars=["Model", "Scenario", "Country code"], 
                                                                                                                var_name="Year", value_name="GDP per capita")

        # 2. Convert GDP per capita to country risk premium 
        collated_results["Country Risk Premium"] = self.country_risk_gdp * collated_results["GDP per capita"] ** self.country_risk

        # 3. Convert GDP per capita to country default spread
        collated_results["Country Default Spread"] = self.cds_gdp * collated_results["GDP per capita"] ** self.cds

        # 3. Convert into long format
        