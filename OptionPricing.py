import pandas as pd
import numpy as np
from scipy.optimize import minimize

class TreeOptionPricing():
    def __init__(self, input_parameters):
        ## input parameters as a dictonary with data being used 
        ## stock_prices - time series of stock prices (pd.Series datatype)
        ## risk_free_rate - is a % that is granted to be paid from any money placed in market's deposit (positive number in %)
        ## expiration_date - is a date after which option is no longer exist (date in same format as in stock_prices time series)
        ## strike_price - fixed price of underlying stock for an option we are looking at (number)
        ## variants - hyperparameter of a model that defines tree structure, number of siblings for each node (integer >= 2)
        ## sigma - is a volatility of stock (derived from it or aprior one)
        ## derive_sigma - flag to know if derive sigma from stock or use aprior one
        ## type - put or call (type of option)
        ## time_step - lenght of each step. Changed to measure in years (so one day will be 1/365)
        
        assert input_parameters['strike_price'] > 0, "Strike price can't be not positive"
        self.strike_price = input_parameters['strike_price']
        
        assert isinstance(input_parameters['risk_free_rate'], float), "Risk-free rate should be a number variable"
        assert input_parameters['risk_free_rate'] > 0, "Risk-free rate should be positive"
        self.risk_free_rate = 1 + input_parameters['risk_free_rate']/100
        
        assert isinstance(input_parameters['number_of_steps'], int), "Number of steps should be an integer variable" 
        assert input_parameters['number_of_steps'] > 1, "Number of steps should be positive and higher than 1"
        self.number_of_steps = input_parameters['number_of_steps']
        
        assert isinstance(input_parameters['stock_prices'], pd.Series), "Stock prices time series should be Series datatype object"
        assert input_parameters['stock_prices'].shape[0] > 0, "Stock prices time series should be not empty"
        self.stock_prices = input_parameters['stock_prices']
        
        assert isinstance(input_parameters['type'], str), "Option type should be string type variable"
        assert input_parameters['type'] in ['call', 'put'], "Option type should be put or call"
        self.type = input_parameters['type']
        
        if input_parameters['derive_sigma']:
            self.sigma = np.std(np.array(self.stock_prices.values))
        else:
            assert isinstance(input_parameters['sigma'], float), "Volatility should be a number variable"
            assert input_parameters['sigma'] > 0, "Volatility should be positive parameter "
            self.sigma = input_parameters['sigma']
        
        assert isinstance(input_parameters['expiration_date'], 
                          type(input_parameters['stock_prices'].index[0])),"Expiration date type is not matching with type in stock prices time series"
        self.expiration_date = input_parameters['expiration_date']
        
        assert isinstance(input_parameters['number_of_variants'], int), "Number of sibling nodes for each node should be an integer"
        assert input_parameters['number_of_variants'] >= 2, "For each node it should be more than one sibling node"
        self.variants = input_parameters['number_of_variants']
        
        assert isinstance(input_parameters['current_date'], 
                                  type(input_parameters['stock_prices'].index[0])),"Current date type is not matching with type in stock prices time series"
        self.current_date = input_parameters['current_date']
        
        time_gap = pd.to_datetime(self.expiration_date) - pd.to_datetime(self.current_date) 
        time_gap = time_gap.days
        
        assert time_gap > 0, "Expiration date of an ption should be after current date"

        self.time_step  = time_gap/(self.number_of_steps*365)
    
    def __arbitrage_equation(self, u):
        # input: u - factor parameter for adjusting stock prices derived from conditions
        # output: value of arbitrage equation absolute difference between right and left side (perfectly should be zero)
        
        # solution from constraints
        solution = self.__system_equations(u)
        
        # factors and probabilities from solution
        factors = solution[1].copy()
        probabilities = solution[0].copy()

        # arbitrage equations
        expectance = np.sum(factors*probabilities)
        risk_free_increment = np.exp(self.risk_free_rate*self.time_step)
        arbitrage_equation = expectance - risk_free_increment
        
        # taking abs(...) and returning result  
        return abs(arbitrage_equation)

        
    def __system_equations(self, u):
        # input: parameter u - for optimization process
        # output: probabilities and factors for that input
        
        # local variable that contains number of variants
        midpoint = self.variants
        
        # initializing current factors (for adjustment of stock price) and probabilities (for each factor adjustment)
        factors = np.zeros(midpoint)
        probabilities = np.zeros(midpoint)
        
        # factors should be degrees of some number u (auxiliary variable from input)
        for i in range(0, int(midpoint/2)):
            factors[i] = u**( int(midpoint/2) -i)
    
        # equations to make tree recombining   
        for i in range(int(midpoint/2)):
            factors[midpoint -1 -i] = 1/factors[i]

        # for case of having 5, 7, 9 ... number of nodes - assigning 1 to middle factor (no changes)
        if midpoint%2 == 1:
            factors[int(midpoint/2)] = 1
        
        # probabilities equation - for each factor - probability of that particular adjustment
        sum_prob = 0
        expanded_list = [*factors, np.exp(self.risk_free_rate*self.time_step)]
        expanded_list.sort()
                                          
        for i in range(0, midpoint):
            probabilities[i] = (expanded_list[i+1]- expanded_list[i])/( factors[0] - factors[midpoint - 1])

        # returning probabilities and factors calculated
        return probabilities, factors
        
        
    def forward_initialization(self):
        # initializing list of factors for stock price used for an every step in a model. Tree should be recombining.
        
        # initializing list of transition probabilities to each of stock prices next step
        self.factors = np.zeros(self.variants)
        self.probabilities = np.zeros(self.variants)
      
        # looking into variants used in model (2 - binomial model, 3 - trinomial model, 4+ -generalized custom model)
        if self.variants ==  2:
            # binomial model parameters initialization
            
            # increment for risk-free investment
            risk_free_increment = np.exp(self.risk_free_rate*self.time_step)
                
            # upscale parameters and temporary parameter u for probability formula 
            self.factors[0] = np.exp(self.sigma * np.sqrt(self.time_step))
            up = np.exp(self.sigma * np.sqrt(self.time_step))
            
            # downscale parameters
            self.factors[1] = 1/up
            down = 1/up
            
            # initializing probabilities
            self.probabilities[0] = (risk_free_increment - down)/(up - down)
            self.probabilities[1] =  1 - self.probabilities[0]
            
        elif self.variants == 3:
            # trinomial model initialization
            
            # upscale factor
            self.factors[0] = np.exp(self.sigma * np.sqrt(2*self.time_step))
            
            # stable factor (no change in stock price)
            self.factors[1] = 1
            
            # downscale factor 
            self.factors[2] = 1/self.factors[0]
            
            # initializing probabilities 
            self.probabilities[0] = (
                (
                    np.exp(self.risk_free_rate*self.time_step/2) - np.exp(- self.sigma * np.sqrt(self.time_step/2))
                )/
                (
                    np.exp(self.sigma * np.sqrt(self.time_step/2)) - np.exp(- self.sigma * np.sqrt(self.time_step/2))
                )
                                    )**2
            
            self.probabilities[2] = (
                (
                    np.exp(self.sigma * np.sqrt(self.time_step/2)) - np.exp(self.risk_free_rate*self.time_step/2)
                )/
                (
                    np.exp(self.sigma * np.sqrt(self.time_step/2)) - np.exp(- self.sigma * np.sqrt(self.time_step/2))
                )
                                    )**2
            
            self.probabilities[1] =  1 - self.probabilities[0] - self.probabilities[2]
        else:
            # general model for more n > 3
            
            # solution that consist from factors and probabilities in that order (+ auxiliary variable that is odd)
            # auxilary variable minimization starts with - 2
            # assigning in x - result of optimization (minimum of arbitrage equation difference minimization)
            x = minimize(self.__arbitrage_equation, 2)['x']
            
            # Initializing probabilities as a result of minimization process
            self.probabilities = self.__system_equations(x[0])[0]
            self.factors = self.__system_equations(x[0])[1]
    
    def __next_layer_build(self, is_first = False, digits_to_round = 4):
        # input:
        # digits_to_round is a accuracy input (number of digits to round)
        # is_first - flag to use if we are initializing first layer
        # output: next layer is added to layers,might be text of that if neccesary
        
        # building stock prices for next layer
        # if that is first layer - just initialize it
        assert isinstance(is_first , bool), " First flag should be boolean"
        if is_first:
            try:
                stock_start = self.stock_prices[self.stock_prices.index == self.current_date][0]
            except:
                raise Exception("Current day is not in stocks time series - check for current date in stocks")

            self.layers = [[stock_start]]
            return "First stock prices layer initialized"
        
        # otherwise - add new layer
        current_layer = np.array( self.layers[-1].copy() )
        next_layer = []
        
        # filling next layer - applying factors to each element using the fact that tree is recombining
        for elem in current_layer:
            next_layer.append(self.factors[0]*elem)
        
        for factor in self.factors[1:]:
            next_layer.append(factor*current_layer[-1])
                
        # checking accuracy input validity
        assert isinstance(digits_to_round, int), "Number of digits to round should be an integer"
        assert digits_to_round >= 0, "Number of digits to round should be not a negative value"
            
        # rounding and sorting
        next_layer = [round(elem, digits_to_round) for elem in next_layer]
        next_layer.sort(reverse = True)

        # asserting that next layer has correct number of elements (as expected)
        assert len(next_layer) == current_layer.shape[0]+self.variants-1, "Next layer number of nodes is unexpected."
        
        # appending a layer and returning a success status
        self.layers.append(next_layer)
        return "Next stock prices layer added"
    
    def tree_build(self, accuracy = 4, vocab = False):
        # input:
        # accuracy is a number of digits to round
        # vocab - boolean variable that enables printing of results
        # output of that function is a complete tree in self.layers
        
        #asserting vocab is a boolean type
        assert isinstance(vocab, bool), "Vocab variable should be boolean type"
        
        # first layer initialized
        text = self.__next_layer_build(is_first = True, digits_to_round = accuracy)
        
        # printing result if enabled
        if vocab:
            print(text)
                
        # number of steps assigned in constructor is used here to define number of layers
        for step in range(self.number_of_steps):

            text = self.__next_layer_build(is_first = False, digits_to_round = accuracy)
            
            # if vocab is enabled - then print information about that
            if vocab:
                print(text)

        if vocab:
            print("Tree initialized for stocks")
            
        return "Tree initialized for stocks"
    
    def __option_price(self, stock_price, option_type, strike_price):
        # input:
        # stock_price - price of underlying asset
        # option_type - call or put - type of an option
        # strike_price - fixed price of stock related to that option -  
        # option gives a right to buy/sell that asset by strike price
        # Output is inner price of an option
        
        # asserting input (same as in constructor)
        assert option_type in ['call', 'put'], "Option type should be \"call\" or \"put\" only"
        assert stock_price >= 0, "Stock price should be positive or zero"
        assert strike_price > 0, "Strike price should be positive"
        
        # returning inner price that depends on type (different for put and call)
        if option_type == 'call':
            return max( stock_price - strike_price, 0)
        elif option_type == 'put':
            return max( strike_price - stock_price, 0)
        
        
    def __next_option_layer_build(self, is_first = False, digits_to_round = 4):
        # input:
        # digits_to_round is a accuracy input (number of digits to round)
        # is_first - indicator of first layer (last one to start with - on expiration date)
        # output: next layer of option's prices is added to layers
        
        # if that is first layer - then initialize layers variable (class field)
        if is_first:
            self.option_layers = [list(map(lambda x: self.__option_price(x, self.type, self.strike_price), self.layers[-1]))]
            return "First option's layer initialized"
            
        # building option prices for previous timestamp
        current_layer = np.array( self.option_layers[-1].copy() )
        layer_size = len(current_layer)
        next_layer = []
        window_size = self.factors.shape[0]

#         if layer_size < window_size:
#             return "There is no more layers to initialize"
        
        # filling next layer
        for i in range(layer_size - window_size+1):
            next_value = self.probabilities@current_layer[i:i + window_size].T*np.exp(-self.risk_free_rate*self.time_step)
            next_layer.append(next_value)
       
        # checking accuracy input
        assert isinstance(digits_to_round, int), "Number of digits to round should be an integer"
        assert digits_to_round >=0, "Number of digits to round should be not negative value"
            
        # rounding and sorting
        next_layer = [round(elem, digits_to_round) for elem in next_layer]
        next_layer.sort()
        
        # appending next layer of option prices and returning text output status
        self.option_layers.append(next_layer)
        return "Next option pricing layer added"
        
    def backward_tree_build(self, accuracy = 4, vocab = False):
        # input:
        # accuracy is a number of digits to round
        # output of that function is a complete tree in self.options_layers with cost function __options_price
        
        # checking vocab is a boolean type
        assert isinstance(vocab, bool), "Vocab variable should be boolean type"
        
        # initializing first option layer (on expiration)
        text = self.__next_option_layer_build(is_first = True, digits_to_round = accuracy)
        
        # printing status if vocab is enabled
        if vocab:
            print(text)
            
        for i in range(self.number_of_steps):
            # adding next step
            text = self.__next_option_layer_build(is_first = False, digits_to_round = accuracy)
            if vocab:
                print(text)
        
        # returning status on process
        if vocab:
            print("Tree initialized for options")
            
        return "Tree initialized for options"
    
    def fit(self, accuracy = 4, vocab= False):
        # input:
        # accuracy - number of digits to round numbers to
        # vocab - boolean variable to look for each stage output
        # output: fitted model with parameters input
        
        # initializing probabilities and factors
        self.forward_initialization()
        
        # build tree for stock prices
        self.tree_build(accuracy = accuracy, vocab = vocab)
        
        # build tree for option prices
        self.backward_tree_build(accuracy = accuracy, vocab = vocab)
    
    def predict(self):
        # output: numerical result of option price predict on fitted model in current date
        return self.option_layers[len(self.option_layers) - 1][0]
                 
    def build_trees(self):
        # method to show stock and options trees a s a tables
        # output: tables for option's and stock's data
        
        result_options = pd.DataFrame(self.option_layers).T
        
        # defining columns names (number of steps)
        cols_stock = ["№"+str(i+1) for i in range(len(result_options.columns))]
        cols_options = ["№"+str(len(result_options.columns) - i) for i in range(len(result_options.columns))]
        
        # writing result for options
        result_options.columns = cols_options
        
        # writing result for stock
        result_stock = pd.DataFrame(self.layers).T
        result_stock.columns = cols_stock
        
        return result_options, result_stock
    
    
