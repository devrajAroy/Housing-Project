import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

filepath = "C:/Users/dev/Documents/Group Project/Housing data 2024.xlsx"

#finding all the data needed
house_prices = pd.read_excel(filepath, sheet_name = 'house prices').iloc[4:175, :]
house_prices.columns = ["years", "house price"]
house_prices = house_prices.reset_index(drop = True) 
interest_rates = pd.read_excel(filepath, sheet_name = 'interest rates').iloc[:513, 2:4]
interest_rates.columns = ["bank interest rate", "mortgage rate"]
interest_rates = interest_rates.reset_index(drop=True)
years = pd.read_excel(filepath, sheet_name = 'retail prices and earnings')
years.columns = ["years", "RPI", "Average Earnings"]
years = years.reset_index(drop = True)

#extrapolating relevant rows making sure only q1 is tskrn from each year
q1house_price = []
for i in range(len(house_prices['years'])):
    if i%4==0 :
        q1house_price.append(house_prices.loc[i,'house price'])

#function to find the mortgage left to pay 
def P_n(mortgage_rate, deposit, N, P_0, n):
    twelve_r = mortgage_rate/100
    r = twelve_r/12
    C = (r/(1-((1+r)**(-N)))) * P_0
    if n > 360:
        P_n = 0
    else: 
        P_n = (((1+r)**n)*P_0)-(((((1+r)**n)-1)/r)*C)
    return P_n

#creating a list with the mortgage left to pay each year
mortgage_to_pay = []
for n in range(len(q1house_price)):
    N = 12*n
    mortgage_to_pay.append(P_n(14.75, 1000, 360, 9000, N))

#creating a list with the equity each year
equity_values = []
for n in range(len(mortgage_to_pay)):
    equity_values.append(q1house_price[n] - mortgage_to_pay[n])  

df = pd.DataFrame(equity_values)
df = df.rename(columns = {0:'equity'})

y_axis = np.linspace(0,200000,20)

#plotting the equity each year
plt.plot(years['years'][1:44], df['equity'])
plt.grid()
plt.xlim(1974,2016)
plt.ylim(0,200000)
plt.xlabel('Year')
plt.ylabel('Equity (£)')
plt.title('Total equity in the house from 1974 to 2016')
plt.show()

#calculating the total monthly payments repaid
r = (14.75/100)/12
C = (r/(1-((1+r)**(-360)))) * 9000
total_monthly_payments = 360*C 
print(total_monthly_payments)

#creating a function to work out salary for a 90% loan
def salary_calculator(x):
   N=x*0.9
   y=N/3
   return y 

#list of salaries for a 90% loan each year
salaries_needed = []
for n in range(len(q1house_price)):
    salaries_needed.append(salary_calculator(q1house_price[n]))
df_salary= pd.Series(salaries_needed)

#plotting salary needed for 90% loan against average salary
plt.plot(years['years'][1:44],df_salary, years['years'][1:44], years['Average Earnings'][1:44])
plt.grid()
plt.xlim(1974,2016)
plt.xlabel('Year')
plt.ylabel('Salary (£)')
plt.title('Average salary compared to salary needed for a 90% loan over time')
plt.legend(labels = ['salary needed for 90% loan','average salary'])
plt.show()

#extrapolating mortgage rates for January each year
mortgage_rates = []
for l in range(len(interest_rates)):
    if l%12==0:
        mortgage_rates.append(interest_rates.loc[l, 'mortgage rate'])
    
#extrapolating bank rates for January each year
bank_rates = []
for u in range(len(interest_rates)):
    if u%12==0:
        bank_rates.append(interest_rates.loc[u, 'bank interest rate'])

#creating function to calculate equity
def calculate_equity(property_value, mortgage_owed):
    equity = property_value - mortgage_owed
    return equity

#taking data from excel and working out relevant prices for each year
df = pd.read_excel(filepath, sheet_name="retail prices and earnings", header=1)
df = df[["Year", "Retail Price Index (2010 = 100)"]]
df["Legal Fees (Buying)"] = 1500
df["Legal Fees (Selling)"] = 400
r, c = df.shape
for i in range(r - 1, -1, -1):
    if i == r - 1:
        continue
    else:
        df.loc[i, "Legal Fees (Buying)"] = int(df.loc[i, "Retail Price Index (2010 = 100)"] / df.loc[
            i + 1, "Retail Price Index (2010 = 100)"] * df.loc[i + 1, "Legal Fees (Buying)"])
        df.loc[i, "Legal Fees (Selling)"] = int(df.loc[i, "Retail Price Index (2010 = 100)"] / df.loc[
            i + 1, "Retail Price Index (2010 = 100)"] * df.loc[i + 1, "Legal Fees (Selling)"])

#creating a function to calculate the fees 
def check_other_fees(property_price, legal_fees):
    extra = 0.003 * property_price
    if property_price < 125000:
        stamp_duty = 0
    elif property_price < 250000:
        stamp_duty = 0.02 * property_price
    elif property_price < 925000:
        stamp_duty = 0.05 * property_price
    elif property_price < 1500000:
        stamp_duty = 0.1 * property_price
    else:
        stamp_duty = 0.12 * property_price
    other_fees = legal_fees + extra + stamp_duty
    return other_fees

years_needed = []
for n in range(1974, 2015, 5):
    years_needed.append(n)

#finding the legal fees when buying a house each year
legal_fees_buying = []
for m in years_needed:
    legal_fees_series = df.loc[df['Year']==m, 'Legal Fees (Buying)']
    legal_fees_individual = int(legal_fees_series.iloc[0]) if not legal_fees_series.empty else 0
    legal_fees_buying.append(legal_fees_individual)

#finding the legal fees when selling a house each year
legal_fees_selling = []
for q in years_needed:
    legal_fees_series = df.loc[df['Year']==q, 'Legal Fees (Selling)']
    legal_fees_individual = int(legal_fees_series.iloc[0]) if not legal_fees_series.empty else 0 
    legal_fees_selling.append(legal_fees_individual)

#taking the house price for relevant years
house_price = []
for i in range(len(q1house_price)):
    if i%5==0 :
        house_price.append(q1house_price[i])

#taking salaries needed for a 90% loan each year
salaries_for_buying = []
for i in range(len(salaries_needed)):
    if i%5==0 :
        salaries_for_buying.append(salaries_needed[i])

#creating a list for all the rpi
rpi = []    
for i in range(len(df['Retail Price Index (2010 = 100)'])):
    rpi.append(df.loc[i,'Retail Price Index (2010 = 100)'])

#listing the mortgage rates in the years of buying a house
mortgage_rates_needed = []
for z in range(len(mortgage_rates)):
    if z%5==0:
        mortgage_rates_needed.append(mortgage_rates[z])

all_of_equity = [1000]
total_investment = [0]

#calculates the rate of inflation each year
inflation_rates = []
for i in range(1, len(rpi)):
    inflation_rate = ((rpi[i]-rpi[i-1])/rpi[i-1])*100
    inflation_rates.append(inflation_rate)


#creating a function that buys and sells a house every 5 years where x is the percentage of equity invested
def whole_function(x):
    
    #creating a function that works out how much is invested after selling
    def investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying):
        y = (1-(x/100))
        return (equity*y) - buying_fee - total_fees_selling - legal_fees_buying
    
    #creating a function that works out the property price able to be bought
    def new_property_price_calculator(E, x, S): #E is equity, S is salary
        property_price = ((E*x)/100) + (3*S)
        return property_price
    
    mortgage_owed = P_n(mortgage_rates_needed[0], 1000, 360, 9000, 60)
    equity = calculate_equity(house_price[1], mortgage_owed) 
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[1]))
    property_prices_bought = [10000]
    property_prices_bought.append(new_property_price)
    
    total = 1
    for num in inflation_rates[:5]:
        total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold = [17793]
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[0]))
    total_fees_selling = legal_fees_selling[0] + 0.004*(house_price[0]) + 0.02*(house_price[0])
    
    invested_money = investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[0]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[1], equity, 360, (3*salaries_for_buying[1]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[2]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[5:10]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[1]))
    total_fees_selling = legal_fees_selling[1] + 0.004*(house_price[1]) + 0.02*(house_price[1])
    
    interest_on_investment = invested_money 
    for num in bank_rates[5:10]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[1]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[2], equity, 360, (3*salaries_for_buying[2]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[3]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[10:15]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[2]))
    total_fees_selling = legal_fees_selling[2] + 0.004*(house_price[2]) + 0.02*(house_price[2])
    
    interest_on_investment = invested_money 
    for num in bank_rates[10:15]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[2]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[3], equity, 360, (3*salaries_for_buying[3]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[4]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[15:20]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[3]))
    total_fees_selling = legal_fees_selling[3] + 0.004*(house_price[3]) + 0.02*(house_price[3])
    
    interest_on_investment = invested_money 
    for num in bank_rates[15:20]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[3]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[4], equity, 360, (3*salaries_for_buying[4]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[5]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[20:25]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[4]))
    total_fees_selling = legal_fees_selling[4] + 0.004*(house_price[4]) + 0.02*(house_price[4])
    
    interest_on_investment = invested_money 
    for num in bank_rates[20:25]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[4]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[5], equity, 360, (3*salaries_for_buying[5]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[6]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[25:30]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[5]))
    total_fees_selling = legal_fees_selling[5] + 0.004*(house_price[5]) + 0.02*(house_price[5])
    
    interest_on_investment = invested_money 
    for num in bank_rates[25:30]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[5]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[6], equity, 360, (3*salaries_for_buying[6]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[7]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[30:35]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[6]))
    total_fees_selling = legal_fees_selling[6] + 0.004*(house_price[6]) + 0.02*(house_price[6])
    
    interest_on_investment = invested_money 
    for num in bank_rates[30:35]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[6]) 
    total_investment.append(invested_money)
    
    mortgage_owed = P_n(mortgage_rates_needed[7], equity, 360, (3*salaries_for_buying[7]), 60)
    equity = calculate_equity(property_price_inflated, mortgage_owed)
    all_of_equity.append(equity)
    
    new_property_price=(new_property_price_calculator(equity, 80, salaries_for_buying[8]))
    property_prices_bought.append(new_property_price)
    
    total = 1 
    for num in inflation_rates[35:40]:
            total *= (1 + num/100)
            
    property_price_inflated = new_property_price*total
    property_prices_sold.append(property_price_inflated)
    
    buying_fee = (check_other_fees(new_property_price, legal_fees_buying[7]))
    total_fees_selling = legal_fees_selling[7] + 0.004*(house_price[7]) + 0.02*(house_price[7])
    
    interest_on_investment = invested_money 
    for num in bank_rates[35:40]:
        interest_on_investment *= (1 + num/100)
    
    invested_money = interest_on_investment + investment_calculator(equity, buying_fee, total_fees_selling, legal_fees_buying[7]) 
    total_investment.append(invested_money)
    
    print(all_of_equity)
    print(property_prices_bought)
    print(property_prices_sold)
    print(total_investment)
    
    return  all_of_equity, total_investment

all_of_equity, total_investment = whole_function(80)

#plotting the equity vs bank investment over years
plt.plot(years_needed,all_of_equity, years_needed,total_investment)
plt.ylim(0,170000)
plt.xlim(1974,2014)
plt.xticks(years_needed)
plt.grid()
plt.title('Value of investments')
plt.xlabel('Years')
plt.ylabel('Money (£)')
plt.legend(labels = ['Equity', 'Bank Investment'])
plt.show()

#plotting the property able to be bought depending on the equity invested in 2016
E = 130000   #setting the equity to £130,000 
S = salaries_needed[42]
print(S)
x = np.linspace(0,100,100)
formula = (E*x/100)+(3*S)
plt.plot(x, formula)
plt.xlim(0,100)
plt.ylim(3*S, 350000)
plt.xlabel('Equity invested in house (%)')
plt.ylabel('Value of the house (£)')
plt.grid()
plt.title('Equity in 2016 depending on equity investeed' )









