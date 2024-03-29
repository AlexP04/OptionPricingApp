# import different modules
import streamlit as st
import pandas as pd
import numpy as np

# import tree option pricing models
from OptionPricing import TreeOptionPricing

# basic webpage settings
st.set_page_config(
    page_title='Option Pricing Models'
)

st.title('Option Pricing Models')

# defining 2 columns - for input and output
col1, col2 = st.columns(spec=[2, 2])


col1.header('Stock information:')
# entering stocks data
try:
    stock_prices = col1.file_uploader('Input stock prices timeseries: ', type=['csv'], key='input_file')
    stock_prices = pd.read_csv(stocks_prices)
    stock_prices = pd.Series(data = list(stock_prices[stock_prices.columns[1]]), index = stock_prices[stock_prices.columns[0]]) 
except:
    # default case - no data entered
    stock_prices = pd.read_csv('default_stock_data.csv')
    stock_prices = pd.Series(data = list(stock_prices[stock_prices.columns[1]]), index = stock_prices[stock_prices.columns[0]]) 

     
# output_stocks = col1.text_input('Output file name (stocks)', value='output_1', key='output_file_1')
derive_sigma = col1.checkbox("Enable sigma derivation from data", value=True)
sigma = col1.number_input('Custom sigma (if preciding option is disabled)', value= 0.0, step=0.1, key='sigma')

col1.header('Option information:')
# options parameters
# output_options = col1.text_input('Output file name (options)', value='output_2', key='output_file_2')
strike_price = col1.number_input('Strike price', value=100.0, step=0.1, key='strike_price')
expiry_date = col1.text_input('Expiry date', value="", key='expiry_date')
option_type = col1.radio('Option type', ['call', 'put'])

col2.header('Market information:')
# market info
risk_free_rate = col2.number_input('Risk free rate (%)', value=2.0, step=0.1, key='risk_free_rate')
current_date = col2.text_input('Current date', value="", key='current_date')

col2.header('Model specification')
# specify number of sibling nodes - 2 = binomial, 3 = trinomaial, 4+ - generalized model.
number_of_variants = col2.number_input('Number of variants for a model (2 - binomial,  3- trinomial, 4+ - generalized model)', value=2, step=1, key='n')
step_number = col2.number_input('Number of steps for a model', value=2, step=1, key='number_of_steps')

params = {
    "stock_prices": stock_prices,
    "strike_price": strike_price,
    "number_of_variants": number_of_variants,
    "risk_free_rate": risk_free_rate,
    "expiration_date": expiry_date,
    'derive_sigma': derive_sigma,
    'sigma': sigma,
    'current_date': current_date,
    'type': option_type,
    'number_of_steps': step_number
}

accuracy = col2.number_input('Accuracy for model (number of digits):', value=2, step=1, key='accuracy')

                        
if col2.button('RUN', key='run'):
    # initializing class
    model = TreeOptionPricing(params)
    
    # fit
    model.fit( accuracy = accuracy)
    
    tree_stocks = pd.DataFrame(model.build_trees()[1])
    tree_options = pd.DataFrame(model.build_trees()[0])
    
    # output - trees for stock and option prices
    st.header("Tree for stock prices")
    st.write(tree_stocks)
    
    st.header("Tree for option prices")
    st.write(tree_options)      
   
    
    #predict for time we are interested for (in 0)
    res = model.predict()
    
    col2.write('Prediction result:' + str(res))
    
        
