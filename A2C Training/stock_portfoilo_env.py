import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 0}
    def __init__(self, obs, price, symbols, initial_fund, both_pos, short_only, long_only, commission_rate, valid_folder, 
                 pos_factor=1, mimimum_share=100, slippage=0.02, commission=0.00015, test=False, valid=False, 
                 risk_free=0.028, active_shares=False, cash_out_stop=True, returns=1, calculate_fund=False, 
                 cashout_freze=False, action_type=1, obs_flatten=False, num_stock=1):
        self.non_increasing_periods = []
        self.long_trades = 0
        self.short_trades = 0
        self.obs_flatten = obs_flatten
        self.action_type = action_type
        self.cashout_freze = cashout_freze
        self.symbols = symbols
        self.pos_factor = pos_factor
        self.active_shares = active_shares
        self.num_stock = num_stock
        self.risk_free = risk_free
        self.slippage = slippage
        self.test = test
        if commission_rate is None:
            self.commission = commission
            self.rate = False
        else:
            self.commission = commission_rate
            self.rate = True
        self.mimimum_share = mimimum_share
        self.both_pos = both_pos
        self.short_only = short_only
        self.long_only = long_only
        if self.both_pos == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(3) # action is [0, 1, 2] 2: short, 1: long, 0: close
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, 3))#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Both Position'
        elif self.long_only == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(2) # action is [0, 1] 1: buy, 0: sell
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, 3))#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Long Only'
        elif self.short_only == True:
            if self.action_type == 1:
                self.action_space = spaces.Discrete(2) # action is [0, 1] 1: short, 0: close
            else:
                self.action_space = spaces.Box(low=0, high=1, shape=(self.num_stock, 3))#Number of RowsIns, Numbers Per Row/Number of Actions
            self.trading_option = 'Short Only'
        self.obs = obs
        self.price = price
        self.obs_index = 0
        if self.obs_flatten == True:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs[0].flatten().shape))
        else:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs[0].shape))
        self.num_long_pos = 0
        self.num_short_pos = 0
        self.accounts = []
        self.account_1 = []
        self.calculate_fund = calculate_fund
        self.max_prices = []
        self.initial_fund_1 = initial_fund
        if self.calculate_fund == True:
            for i in range(self.num_stock):
                try:
                    tmp_array = self.price[:,i].astype('float')
                except:
                    tmp_array = self.price[:,i]
                self.max_prices.append(np.max(tmp_array))
            max_pice = np.max(self.max_prices)
            for i in range(self.num_stock):
                tmp_fund = max_pice*self.mimimum_share/self.pos_factor
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
            print(max_pice, self.mimimum_share, self.pos_factor)
        else:
            for i in range(self.num_stock):
                tmp_fund = initial_fund
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        self.open_price = [None]*self.num_stock
        self.held_pos = [None]*self.num_stock
        self.all_his = []
        self.valid = valid
        self.valid_folder = valid_folder
        self.sharpe_his = [self.initial_fund]
        self.cash_his = [self.initial_fund]
        self.unrealized_profit = [0]*self.num_stock
        self.last_symbol = [None]*self.num_stock
        self.cash_out_stop = cash_out_stop
        self.stop_now = False
        self.returns = returns
        self.current_share = self.mimimum_share
        
    def render(self):
        pass

    def close(self):
        pass

    def MaxDrawdown(self, return_list):
        '''最大回撤率'''
        try:
            i = np.argmax((np.maximum.accumulate(return_list) - return_list)/np.maximum.accumulate(return_list))#结束位置
            if i == 0:
                return 0
            j = np.argmax(return_list[:i])  # 开始位置
            return (return_list[j] - return_list[i]) / (return_list[j])
        except:
            return -np.inf
    def seed(self, seed):
        pass
        
    def sharpe_ratio(self, data, risk_free=0.028):
        try:
            data = pd.Series(data)
            data['return'] = (data-data.iloc[0])/data.iloc[0]
            data = data['return'].values
            sharpe = ((np.mean(data)-risk_free)/np.std(data))#*(252 ** 0.5)
        except:
            sharpe = -np.inf
        return sharpe
    
    def calculate_shares(self, i, tmp_price_1):
        cost = tmp_price_1*self.mimimum_share
        tmp_account_balance = self.accounts[i]
        shares = tmp_account_balance // cost * self.mimimum_share
        return shares
                     
    def step(self, action):
        obs = self.obs[self.obs_index]
        tmp_price = self.price[self.obs_index]
        tic = self.symbols[self.obs_index]
        for i in range(self.num_stock):
            tmp_tic = tic[i]
            try:
                tmp_price_1 = tmp_price[i].astype('float')
            except:
                tmp_price_1 = tmp_price[i]
            if self.action_type == 1:
                tmp_action = action[i]
            else:
                tmp_action = np.argmax(action[i])
            tmp_pos = self.held_pos[i]
            open_price = self.open_price[i]
            if self.short_only == True:
                if tmp_action != 0:
                    tmp_action = 2
                else:
                    pass
            else:
                pass
                
            if self.cashout_freze==True:
                tmp = int(self.accounts[i]/(tmp_price_1*self.mimimum_share))
                self.current_share = tmp
            else:
                pass

            if self.current_share <= 0 and self.cash_out_stop == True:
                self.stop_now = True
            else:
                pass

            #Long
            if tmp_action == 1:
                holding_price = tmp_price_1+self.slippage
                #Open Long
                if tmp_pos == None:
                    if self.active_shares == True:
                        self.current_share = self.calculate_shares(i, tmp_price_1)
                    else:
                        pass
                    self.long_trades += 1
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = self.open_price[i]*self.current_share*self.commission
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'long'
                    self.unrealized_profit[i] = 0
                    self.num_long_pos += 1
                    self.all_his.append(['Function 1-1', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'open long', tmp_price_1, holding_price, self.accounts[i], 
                                        commission, 0])
                    tmp_pos = 'long'

                #Keep Long
                elif tmp_pos == 'long':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = ((tmp_price_1)-open_price)*self.current_share
                    self.accounts[i] += profit
                    self.unrealized_profit[i] = profit
                    self.all_his.append(['Function 1-2', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'hold long', tmp_price_1, open_price, self.accounts[i], 0, 
                                        profit])
                    
                #Close Short and Open Long
                elif tmp_pos == 'short':
                    self.long_trades += 1
                    ################################Close Short############################################
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price-holding_price)*self.current_share
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*self.current_share*self.commission
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    ################################Open Long##############################################
                    if self.active_shares == True:
                        self.current_share = self.calculate_shares(i, tmp_price_1)
                    else:
                        pass
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = self.open_price[i]*self.current_share*self.commission
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'long'
                    self.unrealized_profit[i] = 0
                    self.num_long_pos += 1
                    self.all_his.append(['Function 1-3', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'close short and open long', tmp_price_1, holding_price, 
                                        self.accounts[i], commission, profit])
                    tmp_pos = 'long'
                    
                else:
                    self.all_his.append(['Function 1-4', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'error', tmp_price_1, 0, self.accounts[i], 0, 0])
                
            #Short
            elif tmp_action == 2:
                holding_price = tmp_price_1-self.slippage
                #Open Short
                if tmp_pos == None:
                    if self.active_shares == True:
                        self.current_share = self.calculate_shares(i, tmp_price_1)
                    else:
                        pass
                    self.short_trades += 1
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = self.open_price[i]*self.current_share*self.commission
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'short'
                    self.unrealized_profit[i] = 0
                    self.num_short_pos += 1
                    self.all_his.append(['Function 2-1', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'open short', tmp_price_1, holding_price, self.accounts[i], 
                                        commission, 0])
                    tmp_pos = 'short'

                #Keep Short
                elif tmp_pos == 'short':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price-(tmp_price_1))*self.current_share
                    self.accounts[i] += profit
                    self.unrealized_profit[i] = profit
                    self.all_his.append(['Function 2-2', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'hold short', tmp_price_1, open_price, self.accounts[i], 
                                        0, profit])
                    
                #Close Long and Open Short
                elif tmp_pos == 'long':
                    self.short_trades += 1
                    ################################Close Long############################################
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (holding_price - open_price)*self.current_share
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*self.current_share*self.commission
                    self.accounts[i] += profit
                    self.account_1[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    ################################Open Short##############################################
                    if self.active_shares == True:
                        self.current_share = self.calculate_shares(i, tmp_price_1)
                    else:
                        pass
                    self.open_price[i] = holding_price
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = self.open_price[i]*self.current_share*self.commission
                    self.accounts[i] -= commission
                    self.account_1[i] -= commission
                    self.held_pos[i] = 'short'
                    self.unrealized_profit[i] = 0
                    self.num_short_pos += 1
                    self.all_his.append(['Function 2-3', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'close long and open short', tmp_price_1, holding_price, 
                                        self.accounts[i], commission, profit])
                    tmp_pos = 'short'
                    
                else:
                    self.all_his.append(['Function 2-4', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'error', tmp_price_1, 0, self.accounts[i], 0, 0])

            #Close
            if tmp_action == 0:
                #Close Long
                if tmp_pos =='long':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = ((tmp_price_1 - self.slippage) - open_price)*self.current_share
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = (tmp_price_1 - self.slippage)*self.current_share*self.commission
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    self.held_pos[i] = None
                    self.unrealized_profit[i] = 0
                    self.all_his.append(['Function 3-1', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'close long', tmp_price_1, tmp_price_1 - self.slippage, 
                                        self.accounts[i], commission, profit])
                    tmp_pos = None

                #Close Short
                elif tmp_pos == 'short':
                    self.accounts[i] -= self.unrealized_profit[i]
                    profit = (open_price - (tmp_price_1 + self.slippage))*self.current_share
                    if self.rate == False:
                        commission = self.current_share*self.commission
                    else:
                        commission = (tmp_price_1 + self.slippage)*self.current_share*self.commission
                    self.accounts[i] += profit
                    self.accounts[i] -= commission
                    self.account_1[i] += profit
                    self.account_1[i] -= commission
                    self.held_pos[i] = None 
                    self.unrealized_profit[i] = 0
                    self.all_his.append(['Function 3-2', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'close short', tmp_price_1, tmp_price_1 + self.slippage, 
                                        self.accounts[i], commission, profit])
                    tmp_pos = None
                    
                else:
                    self.all_his.append(['Function 3-3', self.held_pos[i], tmp_action, self.obs_index, i, self.current_share, 
                                        tmp_price[-1], tmp_tic, 'error', tmp_price_1, 0, self.accounts[i], 0, 0])

        self.cash_his.append(np.sum(self.accounts))
        self.sharpe_his.append(np.sum(self.account_1))
        try:
            current_max_index = np.argmax(self.cash_his)
            plains_discount = (self.obs_index - current_max_index)/len(self.obs)
            self.non_increasing_periods.append((self.obs_index - current_max_index))
        except:
            plains_discount = 0
        if self.returns == 1:
            returns = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            maxdrawn = self.MaxDrawdown(self.cash_his)
            reward = returns - abs(maxdrawn)
        elif self.returns == 2:
            reward = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
        elif self.returns == 3:
            returns = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            maxdrawn = self.MaxDrawdown(self.cash_his)
            sharpe = self.sharpe_ratio(self.sharpe_his, self.risk_free)
            reward = ((returns-maxdrawn)*0.5)+(sharpe*0.5)
        else:
            pass

        if self.obs_index == self.obs.shape[0]-1 or self.stop_now == True:
            done = True
            sharpe = self.sharpe_ratio(self.sharpe_his, self.risk_free)
            maxdrawn = self.MaxDrawdown(self.cash_his)
            return_rate = (self.cash_his[-1]-self.initial_fund)/self.initial_fund
            print('------------------This Round Info------------------')
            print('Active Shares: ', self.active_shares, ', Trading Option: ', self.trading_option)
            print('Long Trades: ', self.num_long_pos, ', Short Trades: ', self.num_short_pos)
            print('Step Index: ', self.obs_index)
            print('Initial Account Balance: ', self.initial_fund, ', End Account Balance: ', self.cash_his[-1])
            print('Rate of Return: ', return_rate, ', Sharpe Ratio: ', sharpe)
            print('MaxdrawDown: ', maxdrawn)
            print('Maximum Plain Length: ', np.max(self.non_increasing_periods), 
                      np.max(self.non_increasing_periods)/len(self.obs)*100, '%', 
                      ', Maximum Index: ', np.argmax(self.cash_his))
            print('------------------This Round Ends------------------')
            info = {'done': True, 'return rate':return_rate, 'maxdrawndown': maxdrawn, 'sharpe': sharpe, 
                    'account_history':self.cash_his, 'long trades': self.long_trades, 'short trades': self.short_trades}
        else:
            done = False   
            info = {'done':False, 'return rate':None, 'maxdrawndown': None, 'sharpe': None, 'account_history':None
                    , 'long trades': self.long_trades, 'short trades': self.short_trades}

        if self.valid==True and done==True:
            file_name = str(self.valid_folder+'backtest_'+str(reward)+'_.npy')
            self.cash_history = np.asarray(self.cash_his)
            np.save(file_name, self.cash_his)
            if self.test == True:
                test_df = pd.DataFrame(self.all_his)
                test_df.columns = ['Function Index', 'current holdings', 'model action', 'obs index', 'Loop Index', 
                                   'trading shares', 'Date', 'Ticker', 'action', 'market price', 'holding price', 'cash', 
                                   'commission', 'profit']
                test_df.to_csv(self.valid_folder+'test_df.csv')
        else:
            pass   
        self.obs_index += 1
        if self.obs_flatten == True:
            obs = obs.flatten()
        else:
            pass
        truncated = False
        return obs, reward, done, truncated, info

    def reset(self):
        self.non_increasing_periods = []
        self.long_trades = 0
        self.short_trades = 0
        self.obs_index = 0
        self.num_long_pos = 0
        self.num_short_pos = 0
        self.accounts = []
        self.account_1 = []
        if self.calculate_fund == True:
            for i in range(self.num_stock):
                try:
                    tmp_array = self.price[:,i].astype('float')
                except:
                    tmp_array = self.price[:,i]
                self.max_prices.append(np.max(tmp_array))
            max_pice = np.max(self.max_prices)
            for i in range(self.num_stock):
                tmp_fund = max_pice*self.mimimum_share/self.pos_factor
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
            print(max_pice, self.mimimum_share, self.pos_factor)
        else:
            for i in range(self.num_stock):
                tmp_fund = self.initial_fund_1
                self.accounts.append(tmp_fund)
                self.account_1.append(tmp_fund)
            self.initial_fund = np.sum(self.accounts)
        self.initial_fund = np.sum(self.accounts)
        self.open_price = [None]*self.num_stock
        self.held_pos = [None]*self.num_stock
        self.all_his = []
        self.sharpe_his = [self.initial_fund]
        self.cash_his = [self.initial_fund]
        self.unrealized_profit = [0]*self.num_stock
        obs = self.obs[self.obs_index]
        self.last_symbol = [None]*self.num_stock
        self.stop_now = False
        if self.obs_flatten == True:
            obs = obs.flatten()
        else:
            pass
        self.current_share = self.mimimum_share
        self.obs_index += 1
        info = {}
        return obs, info