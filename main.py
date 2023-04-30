# import different modules
from faulthandler import disable
import streamlit as st
import pandas as pd
import numpy as np

# import tree option pricing models
from OptionPricing import TreeOptionPricing

# basic webpage settings
st.set_page_config(
    page_title='Option Pricing Models',
    layout='wide')

st.title('Option Pricing Models')

# defining 2 columns - for input and output
col1, col2 = st.columns(spec=[2, 2])


st.header('Stock information:')
# entering stocks data
try:
    stocks_prices = col1.file_uploader('Input stock prices timeseries: ', type=['csv'], key='input_file')
    input_df = pd.read_csv(stocks_prices, delimiter=',', decimal='.', index_col=0)
except:
    # default case - no data entered
    stock_prices = pd.read_csv('default_stock_data.csv', delimiter=',', decimal='.', index_col=0)
    st.title('Check input again')
    
derive_sigma = checkbox("Enable sigma derivation from data", value=True)
sigma = col1.number_input('Custom sigma (if preciding option is disabled)', value= 0, step=0.1, key='sigma')


st.header('Option information:')
# options parameters
strike_price = col1.number_input('Strike price', value=100, step=0.1, key='strike_price')
expiry_date = col1.text_input('Expiry date', value="", key='expiry_date')
option_type = col1.radio('Option type', ['call', 'put'])

st.header('Market information:')
# market info
risk_free_rate = col1.number_input('Risk free rate (%)', value=0, step=0.1, key='risk_free_rate')

st.header('Model specification')
# specify number of sibling nodes - 2 = binomial, 3 = trinomaial, 4+ - generalized model.
number_of_variants = col1.number_input('Number of variants for a model (2 - binomial,  3- trinomial, 4+ - generalized model)', value=2, step=1, key='n')

params = {
    "stock_prices": stock_prices,
    "strike_price": strike_price,
    "number_of_variants": number_of_variants,
    "risk_free_rate": risk_free_rate,
    "expiration_date": expiry_date,
    'derive_sigma': derive_sigma,
    'sigma': sigma,
    'time_step': 1,
    'type': option_type
}

accuracy = col1.number_input('Accuracy for model (number of digits):', value=2, step=1, key='accuracy')
agg_method = col1.radio('Aggregation method for option\'s tree', ['mean', 'max', 'min'])
                        
if addon.button('RUN', key='run'):
    # initializing class
    model = TreeOptionPricing(params)
    
    # fit
    model.fit( accuracy = accuracy)
    
    # output - trees for stock and option prices
    
    #predict for time we are interested for
    res = model.predict( T, agg_method)
    
    col2.write('Prediction result:' + str(res))
    
        