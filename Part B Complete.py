import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

filepath = "C:/Users/dev/Documents/Group Project/Housing data 2024.xlsx"

#B1
#extrapolating data from excel for the regions
owner_occupying_region = pd.read_excel(filepath, sheet_name='Type by Region').iloc[3:12, 7:12]
owner_occupying_region.reset_index(drop=True, inplace=True)
location = pd.read_excel(filepath, sheet_name='Type by Region').iloc[3:12, :1]
location.reset_index(drop=True, inplace=True)
years = [1996, 2001, 2006, 2011, 2016]

#creating a graph for the data
plt.figure(figsize=(10, 6))

for index, row in location.iterrows():
    region = row[0]
    region_data = owner_occupying_region.iloc[index]
    plt.plot(years, region_data, marker='o', label=region)

plt.title('Proportion of Household Reference Persons by Region: Owner-Occupying (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc=(0.8, 0.05))
plt.grid(True)

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B1 (a): Trend of the proportion of households in owner-occupying arrangements over time, by region in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.001, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()

#extrapolation data from excel for the countries
owner_occupying_country = pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[13:17, 7:12]
owner_occupying_country.reset_index(drop=True, inplace=True)
country =  pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[13:17, :1]
country.reset_index(drop=True, inplace=True)

#creating the graph to show data
plt.figure(figsize=(10, 6))

for index, row in country.iterrows(): 
    country_name = row[0]
    country_data = owner_occupying_country.iloc[index]
    plt.plot(years, country_data, marker='o', label=country_name)

plt.title('Proportion of Household Reference Persons by Country: Owner-Occupying (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc=(0.8, 0.05))
plt.grid(True)
plt.tight_layout()

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B1 (b): Trend of the proportion of households in owner-occupying arrangements over time, by country in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.001, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()


#B2
#extrapolating data for private renting by region
private_renting_region = pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[21:30, 7:12]
private_renting_region.reset_index(drop=True, inplace=True)

#plotting data
plt.figure(figsize=(10, 6))

for index, row in location.iterrows():
    region = row[0]
    region_data = private_renting_region.iloc[index]
    plt.plot(years, region_data, marker='o', label=region)

plt.title('Proportion of Household Reference Persons by Region: Private Renting (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B2 (a): Trend of the proportion of households that privately rent over time, by region in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.001, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()

#extrapolating data from excel for private renting by country
private_renting_country = pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[31:35, 7:12]
private_renting_country.reset_index(drop=True, inplace=True)

#plotting data
plt.figure(figsize=(10, 6))

for index, row in country.iterrows():
    country_name = row[0]
    country_data = private_renting_country.iloc[index]
    plt.plot(years, country_data, marker='o', label=country_name)

plt.title('Proportion of Household Reference Persons by Country: Private Renting (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B2 (b): Trend of the proportion of households that privately rent over time, by country in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.001, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()

#extrapoltating data for social renting by region
social_renting_region = pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[39:48, 7:12]
social_renting_region.reset_index(drop=True, inplace=True)

#plotting data
plt.figure(figsize=(10, 6))

for index, row in location.iterrows():
    region = row[0]
    region_data = social_renting_region.iloc[index]
    plt.plot(years, region_data, marker='o', label=region)

plt.title('Proportion of Household Reference Persons by Region: Social Renting (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B2 (c): Trend of the proportion of households that socially rent over time, by region in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.001, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()

#extrapolating data for social renting by country
social_renting_country = pd.read_excel(filepath, sheet_name = 'Type by Region').iloc[49:53, 7:12]
social_renting_country.reset_index(drop=True, inplace=True)

#plotting data

plt.figure(figsize=(10, 6))

for index, row in country.iterrows():
    country_name = row[0]
    country_data = social_renting_country.iloc[index]
    plt.plot(years, country_data, marker='o', label=country_name)

plt.title('Proportion of Household Reference Persons by Country: Social Renting (1996-2016)')
plt.xlabel('Year')
plt.ylabel('Proportion (%)')
plt.xticks(years)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

#legend
plt.subplots_adjust(bottom=0.15)
fig_legend_text = 'Figure B2 (d): Trend of the proportion of households that socially rent over time, by country in the UK, every five years from 1996-2016'
plt.figtext(0.5, 0.01, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

plt.show()    


#B3
#exrtapolating data from excel for years, RPI, average earnings and house prices
data = pd.read_excel(filepath, sheet_name = 'retail prices and earnings').iloc[1: , :]
data.columns = ["years", "RPI", "Average Earnings"]
data = data.reset_index(drop = True)

house_prices = pd.read_excel(filepath, sheet_name = 'house prices').iloc[4:175, :]
house_prices.columns = ["years", "house price"]
house_prices = house_prices.reset_index(drop = True)

#only taking average house price from q1 of each year
q1house_price = []
for i in range(len(house_prices['years'])):
    if i%4==0 :
        q1house_price.append(house_prices.loc[i,'house price'])

#turning house prices into a data frame
df = pd.Series(q1house_price)

#plotting RPI across each year
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(data["years"], data["RPI"], 'b-', label='Retail Price Index (2010 = 100)')
ax1.set_xlabel('Year')
ax1.set_ylabel('Retail Price Index', color='black')
ax1.set_xlim([1974,2016])
ax1.tick_params('y', colors='black')
ax1.grid(True)
fig.suptitle('Retail Price Index (1974-2016)')
fig.tight_layout()
plt.show()

#plotting average house price compared to average salary
plt.plot(data["years"], data["Average Earnings"], data["years"],df)
plt.grid()
plt.xlim(1974,2016)
plt.ylim(0,200000)
plt.xlabel('Year')
plt.title("Average UK house price compared to average UK salary")
plt.legend(labels= ['House Price', 'Average Salary'])
plt.show()


#B4
df = pd.read_excel(filepath)
df_sheets = pd.ExcelFile(filepath)
df_region = pd.read_excel(filepath,sheet_name = 'Age group' )
years = df_region.iloc[1,1:6]

horizontal_ticks = list(range(1996, 2017,5))

print(df_region.iloc[2,0:2].isnull().sum())
box = lambda x: df_region.iloc[x,0:2].isnull().sum()
for j in [7]:
    for i in range(2,46):  
            plt.xlabel('Years')
            plt.ylabel(df_region.iloc[0,j])
            plt.xlim(1996,2016)
            plt.xticks(horizontal_ticks)
            plt.grid(True)
            if box(i)==1:
                title=f'{df_region.iloc[i,0]} by age over time'
            elif box(i)==0:
                place= df_region.iloc[i,0]
                people = df_region.iloc[i,j:j+5]
                plt.plot(years,people, marker='o')
                n = i
                while box(n)==0:
                    plt.legend(df_region.iloc[n:i+1,0])
                    n=n-1
               
            else:
                n = i-1
                while box(n)==0:
                    plt.legend(df_region.iloc[n:i,0])
                    n=n-1
                plt.title(title)
                plt.show()
                
              
#B5      
#extrapolate data on ethnicities and owning/renting houses
ethnicities = pd.read_excel(filepath, sheet_name = 'Ethnicity').iloc[3:8, :1]
owner_occupying = pd.read_excel(filepath, sheet_name = 'Ethnicity').iloc[3:8, 6:10]
private_renting = pd.read_excel(filepath, sheet_name = 'Ethnicity').iloc[12:17, 6:10]
social_renting = pd.read_excel(filepath, sheet_name = 'Ethnicity').iloc[21:26, 6:10]
years = [2001, 2006, 2011, 2016]

#plot for owner occupying
plt.figure(figsize=(10, 6))
for i in range(len(ethnicities)):
    plt.plot(years, owner_occupying.iloc[i], label=ethnicities.iloc[i, 0], marker='o')
plt.title('Owner-occupying by Ethnicity')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.xticks(years)
plt.grid(True)
plt.legend()

#legend
fig_legend_text = 'Figure B5 (a): Trend of the proportion of households that are owner-occupied, by ethnicity, every five years from 2001-2016, in the UK.'
plt.figtext(0.5, 0.01, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

#plot for private renting
plt.figure(figsize=(10, 6))
for i in range(len(ethnicities)):
    plt.plot(years, private_renting.iloc[i], label=ethnicities.iloc[i,0], marker='o')
plt.title('Private Renting by Ethnicity')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.xticks(years)
plt.grid(True)
plt.legend()

#legend
fig_legend_text = 'Figure B5 (b): Trend of the proportion of households that privately rent over time, by ethnicity, every five years from 2001-2016, in the UK.'
plt.figtext(0.5, 0.01, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)

#plot for social renting
plt.figure(figsize=(10, 6))
for i in range(len(ethnicities)):
    plt.plot(years, social_renting.iloc[i], label=ethnicities.iloc[i,0], marker='o')
plt.title('Social Renting by Ethnicity')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.xticks(years)
plt.grid(True)
plt.legend(loc=(0.7,0.7))

#legend
fig_legend_text = 'Figure B5 (c): Trend of the proportion of households that socially rent over time, by ethnicity, every five years from 2001-2016, in the UK.'
plt.figtext(0.5, 0.01, fig_legend_text, wrap=True, horizontalalignment='center', fontsize=10)
plt.show()


#B6
# By country regression lines for ownership

data = pd.read_excel(filepath, sheet_name='Type by Region')

years = np.array([1996, 2001, 2006, 2011, 2016])
additional_years = np.linspace(years.min(), years.max() + 10, 100)

countries_data = data.iloc[13:17, 7:12]
country_names = data.iloc[13:17, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]

    poly5 = PolynomialFeatures(degree=5, include_bias=False)

    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))

    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)

    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')

    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Owner-occupying households by country")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By country regression lines for private renting
countries_data = data.iloc[31:35, 7:12]
country_names = data.iloc[13:17, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))

    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)

    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')

    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Private renting by country")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By country regression lines for social renting
countries_data = data.iloc[49:53, 7:12]
country_names = data.iloc[13:17, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    # Fit polynomial regression models
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    # Fit transform for original years
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    # Transform for additional years
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    # Predict values for both original and additional years
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    # Plot the line using both original and additional years
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Social renting by country")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# RPI regression line
data = pd.read_excel(filepath, sheet_name='retail prices and earnings')

years = data.iloc[1:44, 0].values
values = data.iloc[1:44, 1].values

plt.figure(figsize=(10, 6))

poly1 = PolynomialFeatures(degree=3, include_bias=False)

poly1_ind_var = poly1.fit_transform(np.array(years).reshape(-1, 1))

poly_reg1 = LinearRegression(fit_intercept=True)
poly_reg1.fit(poly1_ind_var, values)

y_predicted_1 = poly_reg1.predict(poly1_ind_var)

plt.scatter(years, values, label='Data Points')
plt.plot(years, y_predicted_1, color='red', label='Cubic regression Line')

plt.title("RPI")
plt.xlabel('Years')
plt.ylabel('Values')
plt.grid(True)
plt.legend(fontsize='large')
plt.show()

# By age regression lines for ownership
data = pd.read_excel(filepath, sheet_name='Age group')

years = np.array([1996, 2001, 2006, 2011, 2016])
additional_years = np.linspace(years.min(), years.max() + 10, 100)

countries_data = data.iloc[11:14, 7:12]
country_names = data.iloc[11:14, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]

    poly5 = PolynomialFeatures(degree=5, include_bias=False)

    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Owner occupying by age group")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By age regression lines for private renting
countries_data = data.iloc[26:29, 7:12]
country_names = data.iloc[26:29, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Private renting by age group")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By age regression lines for social renting
countries_data = data.iloc[41:44, 7:12]
country_names = data.iloc[41:44, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Social renting by age group")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By ethnicity regression lines for ownership
data = pd.read_excel(filepath, sheet_name='Ethnicity')

years = np.array([2001, 2006, 2011, 2016])

countries_data = data.iloc[3:8, 6:10]
country_names = data.iloc[3:8, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Owner occupying by ethnicity")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By ethnicity regression lines for private renting
countries_data = data.iloc[12:17, 6:10]
country_names = data.iloc[12:17, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Private renting by ethnicity")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()

# By ethnicity regression lines for social renting
countries_data = data.iloc[21:26, 6:10]
country_names = data.iloc[21:26, 0].values

plt.figure(figsize=(10, 6))
for i in range(countries_data.shape[0]):
    country_data = countries_data.iloc[i]
    country_name = country_names[i]
   
    poly5 = PolynomialFeatures(degree=5, include_bias=False)
   
    poly5_ind_var = poly5.fit_transform(np.array(years).reshape(-1, 1))
   
    poly5_additional_ind_var = poly5.fit_transform(additional_years.reshape(-1, 1))
   
    poly_reg5 = LinearRegression(fit_intercept=True)
    poly_reg5.fit(poly5_ind_var, country_data)
   
    y_predicted_5 = poly_reg5.predict(poly5_ind_var)
    y_predicted_additional_5 = poly_reg5.predict(poly5_additional_ind_var)

    plt.scatter(years, country_data, label=f'{country_name}')
   
    plt.plot(additional_years, y_predicted_additional_5, label=f'{country_name} Quintic Regression')
   

plt.title("Social renting by ethnicity")
plt.xlabel('Years')
plt.ylabel('Proportion of household reference persons')
plt.xticks(years)
plt.grid(True)
plt.legend(fontsize='small')
plt.show()



#B7
# 1) Comparison of home-ownership and renting between UK born and Non-UK born people.
# 2) Relationship between interest rates and mortgage rates on house prices overtime.
# 3) How changes in mortgage rates and interests rates affects home ownership and renting. 

# Data for Owner-occupying tenure
years = ['1996', '2001', '2006', '2011', '2016']
categories = ['UK born', 'Not UK born']
values = np.array([[68, 71, 72, 68, 68],
                   [59, 58, 51, 44, 45]])

# Plot for Owner-occupying
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
r1 = np.arange(len(years))
for i, category in enumerate(categories):
    ax.bar(r1 + i * bar_width, values[i], label=f'{category}', width=bar_width, edgecolor='grey')
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of household reference persons')
ax.set_title('Owner-occupying Tenure by Country of Birth')
ax.set_xticks([r + bar_width / 2 for r in range(len(years))], years)
ax.legend()
plt.show()

# Data for Private renting tenure
years = ['1996', '2001', '2006', '2011', '2016']
categories = ['UK born', 'Not UK born']
values = np.array([[8, 7, 9, 13, 14],
                   [17, 17, 27, 34, 35]])

# Plot for Private renting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
r1 = np.arange(len(years))
for i, category in enumerate(categories):
    ax.bar(r1 + i * bar_width, values[i], label=f'{category}', width=bar_width, edgecolor='grey')
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of household reference persons')
ax.set_title('Private renting Tenure by Country of Birth')
ax.set_xticks([r + bar_width / 2 for r in range(len(years))], years)
ax.legend()
plt.show()

# Data for Social renting tenure
years = ['1996', '2001', '2006', '2011', '2016']
categories = ['UK born', 'Not UK born']
values = np.array([[23, 20, 18, 17, 17],
                   [22, 23, 20, 20, 19]])

# Plot for Social renting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
r1 = np.arange(len(years))
for i, category in enumerate(categories):
    ax.bar(r1 + i * bar_width, values[i], label=f'{category}', width=bar_width, edgecolor='grey')
ax.set_xlabel('Year')
ax.set_ylabel('Proportion of household reference persons')
ax.set_title('Social renting Tenure by Country of Birth')
ax.set_xticks([r + bar_width / 2 for r in range(len(years))], years)
ax.legend()
plt.show()

# Data for Owner-occupying tenure
years = ['1996', '2001', '2006', '2011', '2016']
categories = ['Ownwer occupying', 'Private renting', 'Social Renting']
values = np.array([[67, 71, 69, 65, 64],
                   [9, 9, 11, 16, 17], [23, 21, 18, 18, 17]])

# Plot for all housing types 
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2  
r = np.arange(len(years))  
for i, category in enumerate(categories):
    ax.bar(r + bar_width * i, values[i], label=f'{category}', width=bar_width, edgecolor='grey')

ax.set_xlabel('Year')
ax.set_ylabel('Proportion of household reference persons')
ax.set_title('Total proportion of each housing type')
ax.set_xticks(r + bar_width * (len(categories) - 1) / 2) 
ax.set_xticklabels(years)
ax.legend()

plt.show()

# Data
house_prices = pd.read_excel(filepath, sheet_name = 'house prices').iloc[4:175, :]
house_prices.columns = ["years", "house price"]
house_prices = house_prices.reset_index(drop = True)
years = pd.read_excel(filepath, sheet_name = 'retail prices and earnings')
years.columns = ["years", "RPI", "Average Earnings"]
years = years.reset_index(drop = True)
interest_rates = pd.read_excel(filepath, sheet_name = 'interest rates').iloc[:513, 2:4]
interest_rates.columns = ["bank interest rate", "mortgage rate"]
interest_rates = interest_rates.reset_index(drop=True)

q1house_price = []
for i in range(len(house_prices['years'])):
    if i%4==0 :
        q1house_price.append(house_prices.loc[i,'house price'])
        
mortgage_rates = []
for l in range(len(interest_rates)):
    if l%12==0:
        mortgage_rates.append(interest_rates.loc[l, 'mortgage rate'])
        
bank_rates = []
for u in range(len(interest_rates)):
    if u%12==0:
        bank_rates.append(interest_rates.loc[u, 'bank interest rate'])

# Plot
fig, ax1 = plt.subplots()

# Plot for  interest rates
color = 'black'
ax1.set_xlabel('Year', color=color)
ax1.set_ylabel('Interest Rate (%)', color=color)
ax1.plot(years['years'][1:44], bank_rates, color='red', label='Interest Rate (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.yaxis.label.set_color(color)

# Plot for  mortgage rates
ax1.plot(years['years'][1:44], mortgage_rates, color='blue', label='Mortgage Rate (%)')
ax1.tick_params(axis='y', colors=color)

# Plot for house prices
color = 'black'
ax2 = ax1.twinx()
ax2.set_ylabel('House Price (Â£)', color=color)
ax2.plot(years['years'][1:44], q1house_price, color='green', label='House Price (Â£)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.yaxis.label.set_color(color)

# legends
fig.tight_layout()
fig.legend(loc='lower left', bbox_to_anchor=(0.1, 0.2))

plt.title('Interest Rates vs Mortgage Rates vs House Prices over Time', color=color)
plt.show()







