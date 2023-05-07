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
        
        assert input_parameters['risk_free_rate'] > 0, "Risk-free rate should be positive"
        self.risk_free_rate = 1 + input_parameters['risk_free_rate']/100
        
       
        assert isinstance(input_parameters['stock_prices'], pd.Series), "Stock prices time series should be Series datatype"
        assert input_parameters['stock_prices'].shape[0] > 0, "Stock prices time series should be not empty"
        self.stock_prices = input_parameters['stock_prices']
        
        assert input_parameters['type'] in ['call', 'put'], "Option type should be put or call"
        self.type = input_parameters['type']
        
        if input_parameters['derive_sigma']:
            self.sigma = np.std(np.array(self.stock_prices.values))
        else:
            assert input_parameters['sigma'] > 0, "Volatility should be positive parameter "
            self.sigma = input_parameters['sigma']
        
        assert isinstance(input_parameters['expiration_date'], 
                          type(input_parameters['stock_prices'].index[0])),"Expiration date type is not matching with type in stock prices time series"
        self.expiration_date = input_parameters['expiration_date']
        
        assert isinstance(input_parameters['number_of_variants'], int), "Number of sibling nodes for each node should be an integer"
        assert input_parameters['number_of_variants'] >= 2, "For each node it should be more than one sibling node"
        self.variants = input_parameters['number_of_variants']
        
        assert input_parameters['time_step'] > 0, "Time step that is used should be positive"
        self.time_step  = input_parameters['time_step']/365
    
    def __arbitrage_equation(self, u):
        
        # solution from constraints
        solution = self.__system_equations(u)
        
        # factors and probabilities from solution
        factors = solution[1].copy()
        probabilities = solution[0].copy()

        # arbitrage equations
        expectance = np.sum(factors*probabilities)
        risk_free_increment = np.exp(self.risk_free_rate*self.time_step)
        arbitrage_equation = expectance - risk_free_increment
        
        # taking abs(...)    
        return abs(arbitrage_equation)

        
    def __system_equations(self, u):
        # just local variable that contains number of variants
        midpoint = self.variants
        
        # init current factors (for adjustment of stock price) and probabilities (for each factor adjustment)
        factors = np.zeros(midpoint)
        probabilities = np.zeros(midpoint)
        
        # factors should be degrees of some number u (auxiliary variable)
        for i in range(0, int(midpoint/2)):
            factors[i] = u**( int(midpoint/2) -i)
    
        # equations to make tree recombining   
        for i in range(int(midpoint/2)):
            factors[midpoint -1 -i] = 1/factors[i]

        if midpoint%2 == 1:
            factors[int(midpoint/2)] = 1
        
        sum_prob = 0
        # probabilities equation
        expanded_list = [*factors, np.exp(self.risk_free_rate*self.time_step)]
        expanded_list.sort()
                                          
        for i in range(0, midpoint):
            probabilities[i] = (expanded_list[i+1]- expanded_list[i])/( factors[0] - factors[midpoint - 1])

        return probabilities, factors
        
        
    def forward_initialization(self):
        ## initializing list of factors for stock price used for an every step in a model. Tree should be recombining.
        ## initializing list of transition probabilities to each of stock prices next step
        self.factors = np.zeros(self.variants)
        self.probabilities = np.zeros(self.variants)
      
        if self.variants ==  2:
            # binomial model parameters initialization
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
            x = minimize(self.__arbitrage_equation, 2)['x']
            
            self.probabilities = self.__system_equations(x[0])[0]
            self.factors = self.__system_equations(x[0])[1]
    
    def __next_layer_build(self, is_first = False, digits_to_round = 4):
        # digits_to_round is a accuracy input (number of digits to round)
        # is_first - flag to use if we are initializing first layer
        # output - next layer is added to layers
        
        # building stock prices for next layer
        # if that is first layer - just initialize it
        assert isinstance(is_first , bool), " First flag should be boolean"
        
        if is_first:
            stock_start = self.stock_prices[0]
            
            self.layers = [[stock_start]]
            return "First layer initialized"
        
        # otherwise - add new layer
        current_layer = np.array( self.layers[-1].copy() )
        next_layer = []
        
        # filling next layer
        for elem in current_layer:
            next_layer.append(self.factors[0]*elem)
        
        for factor in self.factors[1:]:
            next_layer.append(factor*current_layer[-1])
                
        # checking accuracy input
        assert isinstance(digits_to_round, int), "Number of digits to round should be an integer"
        assert digits_to_round >=0, "Number of digits to round should be not negative value"
            
        # rounding and sorting
        next_layer = [round(elem, digits_to_round) for elem in next_layer]
        next_layer.sort(reverse = True)

        assert len(next_layer) == current_layer.shape[0]+self.variants-1, "Next layer number of nodes is unexpected."
        
        self.layers.append(next_layer)

        return "Next layer added"
    
    def tree_build(self, accuracy = 4, vocab = False):
        # accuracy is a number of digits to round
        # vocan - boolean variable that enables printing
        # result of that function is a complete tree in self.layers
        
        assert isinstance(vocab, bool), "Vocab variable should be boolean type"
        
        text = self.__next_layer_build(is_first = True, digits_to_round = accuracy)
        
        iteration_list = list(set(self.stock_prices.index[1:]))
        iteration_list.sort()
        
        for date in iteration_list:
            if vocab:
                print(text)
                
            if date > self.expiration_date:
                if vocab:
                    print("Tree initialized for stocks")
                return "Tree initialized for stocks"
            
            text = self.__next_layer_build(is_first = False, digits_to_round = accuracy)
            print(self.layers)
        if vocab:
            print("Tree initialized for stocks, option expiry date is out of stock history present")
            
        return "Tree initialized for stocks, option expiry date is out of stock history present"
    
    def __option_price(self, stock_price, option_type, strike_price):
        # stock_price - price of underlying price
        # option_type - call or put - type of an option
        # strike_price - fixed price of stock related to that option. Option gives a right to buy/sell that asset by strike price
        # Output is inner price of an option
        
        assert option_type in ['call', 'put'], "Option type should be \"call\" or \"put\" only"
        assert stock_price >= 0, "Stock price should be positive or zero"
        assert strike_price > 0, "Strike price should be positive"
        
        if option_type == 'call':
            return max( stock_price - strike_price, 0)
        elif option_type == 'put':
            return max( strike_price - stock_price, 0)
        
        
    def __next_option_layer_build(self, is_first = False, digits_to_round = 4):
        # digits_to_round is a accuracy input (number of digits to round)
        # is_first - indicator of first layer (last one to start with)
        # output - next layer is added to layers
        
        # if that is first layer - then initialize layers variable (class field)
        if is_first:
            self.option_layers = [list(map(lambda x: self.__option_price(x, self.type, self.strike_price), self.layers[-1]))]
            return "First options layer initialized"
            
        # building option prices for previous timestamp
        current_layer = np.array( self.option_layers[-1].copy() )
        layer_size = len(current_layer)
        next_layer = []
        window_size = self.factors.shape[0]

        if layer_size < window_size:
            return "There is no more layers to initialize"
        
        # filling next layer
        for i in range(layer_size - window_size+1):
            next_value = self.probabilities@current_layer[i:i + window_size].T*np.exp(-self.risk_free_rate*self.time_step)
            next_layer.append(next_value)
       
        # checking accuracy input
        assert isinstance(digits_to_round, int), "Number of digits to round should be an integer"
        assert digits_to_round >=0, "Number of digits to round should be not negative value"
            
        # rounding and sorting
        next_layer = [round(elem, digits_to_round) for elem in next_layer]

        
        self.option_layers.append(next_layer)

        return "Next option pricing layer added"
        
    def backward_tree_build(self, accuracy = 4, vocab = False):
        # accuracy is a number of digits to round
        # result of that function is a complete tree in self.options_layers with cost function __options_price
        
        assert isinstance(vocab, bool), "Vocab variable should be boolean type"
        
        text = self.__next_option_layer_build(is_first = True, digits_to_round = accuracy)
            
        for date in list(self.stock_prices.index[1:]):
            if vocab:
                print(text)
                
            text = self.__next_option_layer_build(is_first = False, digits_to_round = accuracy)
        
        if vocab:
            print("Tree initialized for options")
            
        return "Tree initialized for options"
    
    def fit(self, accuracy = 4, vocab= False):
        # initializing probabilities and factors
        self.forward_initialization()
        
        # build tree for stock prices
        self.tree_build(accuracy = accuracy, vocab = vocab)
        
        # build tree for option prices
        self.backward_tree_build(accuracy = accuracy, vocab = vocab)
    
    def predict(self, timestamp, aggregation_method = 'max'):
        # timestamp - time at which option price we are looking for (need to have fit function implemented)
        
        max_index = list(self.stock_prices.index).index(max(self.stock_prices.index))
        expr_index = list(self.stock_prices.index).index(self.expiration_date)
           
        assert timestamp <= max_index, "Timestamp given should be lower than maximum date avaliaible"
        assert timestamp <= expr_index, "Timestamp input should be less or equal to expiration date"
        assert aggregation_method in ['mean', 'max', 'min'], "Aggregation method should be in following list: \'mean\', \'max\', \'min\' "
        
        timestamp = len(self.option_layers) - 1 - timestamp

        print(timestamp)
        print()
        if aggregation_method == 'mean':
            return np.mean(self.option_layers[timestamp])
        elif aggregation_method == 'max':
            return max(self.option_layers[timestamp])
        elif aggregation_method == 'min':
            return min(self.option_layers[timestamp])
        
    
    def build_trees(self):
        result_options = pd.DataFrame(self.option_layers).T
        
        cols_stock = ["№"+str(i+1) for i in range(len(result_options.columns))]
        cols_options = ["№"+str(len(result_options.columns) - i) for i in range(len(result_options.columns))]

        result_options.columns = cols_options
        
        result_stock = pd.DataFrame(self.layers).T
        result_stock.columns = cols_stock
        
        return result_options, result_stock
    
    
