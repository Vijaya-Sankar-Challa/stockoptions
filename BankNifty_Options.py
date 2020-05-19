#!/usr/bin/env python
import requests
import json
import pandas as pd
from time import sleep
from datetime import datetime, time, timedelta, date
import os
import numpy as np
from math import log, sqrt, pi, exp
from scipy.stats import norm


# Underlying price (per share): S;
# Strike price of the option (per share): K;
# Time to maturity (years): T;
# Continuously compounding risk-free interest rate: r;
# Volatility: sigma;

## define two functions, d1 and d2 in Black-Scholes model
def d1(S,K,T,r,sigma):
    #n1 = log(S/K)
    #n2 = (r+sigma**2/2.)*T
    #print("T = ",T)
    #de1 = sqrt(T)
    #return (n1+n2)/sigma*de1
    return(log(S/K)+(r+sigma**2/2.)*T)/sigma*sqrt(T)
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*sqrt(T)

## define the call options price function
def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))

## define the put options price function
def bs_put(S,K,T,r,sigma):
    return K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)

## define the Call_Greeks of an option
def call_delta(S,K,T,r,sigma):
    return norm.cdf(d1(S,K,T,r,sigma))
def call_gamma(S,K,T,r,sigma):
    return norm.pdf(d1(S,K,T,r,sigma))/(S*sigma*sqrt(T))
def call_vega(S,K,T,r,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,sigma))*sqrt(T))
def call_theta(S,K,T,r,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,sigma))*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma)))
def call_rho(S,K,T,r,sigma):
    return 0.01*(K*T*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma)))

## define the Put_Greeks of an option
def put_delta(S,K,T,r,sigma):
    return -norm.cdf(-d1(S,K,T,r,sigma))
def put_gamma(S,K,T,r,sigma):
    return norm.pdf(d1(S,K,T,r,sigma))/(S*sigma*sqrt(T))
def put_vega(S,K,T,r,sigma):
    return 0.01*(S*norm.pdf(d1(S,K,T,r,sigma))*sqrt(T))
def put_theta(S,K,T,r,sigma):
    return 0.01*(-(S*norm.pdf(d1(S,K,T,r,sigma))*sigma)/(2*sqrt(T)) + r*K*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))
def put_rho(S,K,T,r,sigma):
    return 0.01*(-K*T*exp(-r*T)*norm.cdf(-d2(S,K,T,r,sigma)))

## to calculate the volatility of a put/call option

def implied_volatility(Price,S,K,T,r,option):
    sigma = 0.001
    #print (np.array([['Price', 'S', 'K', 'T', 'r'], [Price, S, K, T, r]]))
    if 'CE' in option:
        while sigma < 1:
            Price_implied = S*norm.cdf(d1(S,K,T,r,sigma))-K*exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
            if Price-(Price_implied) < 0.001:
                return sigma*0.95
            sigma += 0.001
        #print("It could not find the right volatility of the call option.",sigma)
    else:
        while sigma < 1:
            Price_implied = K*exp(-r*T)-S+bs_call(S,K,T,r,sigma)
            if Price-(Price_implied) < 0.001:
                return sigma*0.95
            sigma += 0.001
        #print("It could not find the right volatility of the put option.",sigma)
    return sigma


#  Functions that return d_1, d_2 and call and put prices
def d(sigma, S, K, r, t):
    try:
        d1 = 1 / (sigma * np.sqrt(t)) * (np.log(S / K) + (r + sigma ** 2 / 2) * t)
        d2 = d1 - sigma * np.sqrt(t)
        return d1, d2
    except Exception as e:
        print('Error Occured While calculating D:', e)
        print("Underlying:", S, " Strike:", K, " Time:", t, " Volitality:", sigma)
        return

def call_price(sigma, S, K, r, t, d1, d2):
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * t)
    return C

def put_price(sigma, S, K, r, t, d1, d2):
    P = -norm.cdf(-d1) * S + norm.cdf(-d2) * K * np.exp(-r * t)
    return P

#  Functions for Deltam Gamma, and  Theta
def delta(d_1, contract_type):
    if contract_type == 'c':
        return norm.cdf(d_1)
    if contract_type == 'p':
        return -norm.cdf(-d_1)


def gamma(d2, S, K, sigma, r, t):
    return (K * np.exp(-r * t) * (norm.pdf(d2) / (S ** 2 * sigma * np.sqrt(t))))


def theta(d1, d2, S, K, sigma, r, t, contract_type):
    if contract_type == 'c':
        theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    if contract_type == 'p':
        theta = -S * sigma * norm.pdf(-d1) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)

    return theta

def dir_contents():
    file1 = open("filenames.txt", "w")
    for i in os.listdir("Data"):
        file1.write(i + "\n")
    file1.close()

pd.set_option('display.width', 1500)
pd.set_option('display.max_columns', 75)
pd.set_option('display.max_rows', 1500)

r = 10/100 # Compound risk Rate
n_url = 'https://nseindia.com/api/option-chain-indices?symbol=NIFTY'
b_url = 'https://nseindia.com/api/option-chain-indices?symbol=BANKNIFTY'
df_list = []
dir_write = 0

oi_filename = os.path.join(".", "Data/banknifty_{0}.json".format(datetime.now().strftime("%d%m%y")))

expiry = ''
def fetch_oi(df):
    print("Sending Url")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
               "accept-encoding": 'gzip, deflate', "accept-language": 'en-GB,en-US;q=0.9,en;q=0.8'}
    while(1):
        try:
            nifty_url = requests.get(b_url, headers=headers).json()
            #banknifty_url = requests.get(b_url, headers=headers).json()
            break
        except Exception as err:
            print("Error Reading URL: ",err)
            continue

    print("Url Fetched")

    try:
        if expiry:
            ce_values = [data['CE'] for data in nifty_url['filtered']['data'] if "CE" in data and str(nifty_url['expiryDates']).lower() == str(expiry).lower()]
            pe_values = [data['PE'] for data in nifty_url['filtered']['data'] if "PE" in data and str(nifty_url['expiryDates']).lower() == str(expiry).lower()]
        else:
            ce_values = [data['CE'] for data in nifty_url['filtered']['data'] if "CE" in data]
            pe_values = [data['PE'] for data in nifty_url['filtered']['data'] if "PE" in data]
    except Exception as e:
        print("Error while reading JSON: ",e)
        return pd.DataFrame()

    with open("Data/Databanknifty.json", 'w') as files:
        files.write(json.dumps(nifty_url, indent=4, sort_keys=True))
    print("Written to Json File")

    print(" =>Calculating CALL Greeks")
    for i in range(0, len(ce_values)):
        S = ce_values[i]['underlyingValue']
        K = ce_values[i]['strikePrice']
        expiration_date = datetime.strptime(ce_values[i]['expiryDate'], "%d-%b-%Y")
        T = abs(((expiration_date - datetime.utcnow()).days + 1)/ 365)
        #print("===>Time Days:",(expiration_date - datetime.utcnow()).days + 1)
        if T == 0:
            T = 0.0005
        sigma = ce_values[i]['impliedVolatility'] / 100
        t = T
        if sigma == 0:
            sigma = 0.001
        d1, d2 = d(sigma, S, K, r, t)
        #ce_values[i]['delta_t'] = round(call_delta(S, K, T, r, sigma), 2)
        ce_values[i]['delta_t'] = round(delta(d1, 'c'), 2)
        #ce_values[i]['gamma_t'] = round(call_gamma(S, K, T, r, sigma), 2)
        ce_values[i]['gamma_t'] = round(gamma(d2, S, K, sigma, r, t), 4)
        ce_values[i]['vega_t'] = round(call_vega(S, K, T, r, sigma), 2)
        #ce_values[i]['theta_t'] = round(call_theta(S, K, T, r, sigma), 2)
        ce_values[i]['theta_t'] = round(theta(d1, d2, S, K, sigma, r, t, 'c')/365, 2)
        ce_values[i]['rho_t'] = round(call_rho(S, K, T, r, sigma), 2)
        ce_values[i]['price_t'] = round(bs_call(S, K, T, r, sigma), 2)
        ce_values[i]['impliedVolatility_t'] = round(
            100 * float(implied_volatility(ce_values[i]['lastPrice'], S, K, T, r, 'CE')), 2)

    print(" =>Calculating PUT Greeks")
    for i in range(0, len(pe_values)):
        S = pe_values[i]['underlyingValue']
        K = pe_values[i]['strikePrice']
        expiration_date = datetime.strptime(pe_values[i]['expiryDate'], "%d-%b-%Y")
        T = abs(((expiration_date - datetime.utcnow()).days + 1) / 365)
        if T == 0:
            T = 0.0005
        sigma = pe_values[i]['impliedVolatility'] / 100
        t = T
        if sigma == 0:
            sigma = 0.001
        d1, d2 = d(sigma, S, K, r, t)
        #pe_values[i]['delta_t'] = round(put_delta(S, K, T, r, sigma), 2)
        pe_values[i]['delta_t'] = round(delta(d1, 'p'), 2)
        #pe_values[i]['gamma_t'] = round(put_gamma(S, K, T, r, sigma), 2)
        pe_values[i]['gamma_t'] = round(gamma(d2, S, K, sigma, r, t), 4)
        pe_values[i]['vega_t'] = round(put_vega(S, K, T, r, sigma), 2)
        #pe_values[i]['theta_t'] = round(put_theta(S, K, T, r, sigma), 2)
        pe_values[i]['theta_t'] = round(theta(d1, d2, S, K, sigma, r, t, 'p')/365, 2)
        pe_values[i]['rho_t'] = round(put_rho(S, K, T, r, sigma), 2)
        pe_values[i]['price_t'] = round(bs_put(S, K, T, r, sigma), 2)
        pe_values[i]['impliedVolatility_t'] = round(
            100 * float(implied_volatility(pe_values[i]['lastPrice'], S, K, T, r, 'PE')), 2)

    print(" =>Completed Calculations")

    ce_data = pd.DataFrame(ce_values).sort_values(['strikePrice'])
    pe_data = pd.DataFrame(pe_values).sort_values(['strikePrice'])

    ce_data['type'] = "CE"
    pe_data['type'] = "PE"

    df1 = pd.concat([ce_data, pe_data])

    if len(df_list) > 0:
        df1['Time'] = df_list[-1][0]['Time']
    if len(df_list) > 0 and df1.to_dict('records') == df_list[-1]:
        print("Duplicate data found, Refreshing after 1 min")
        sleep(60)
        return pd.DataFrame()
    df1['Time'] = datetime.now().strftime("%H:%M")

    if not df.empty:
        df = df[
            ['strikePrice', 'expiryDate', 'underlying', 'identifier', 'openInterest', 'changeinOpenInterest',
            'pchangeinOpenInterest', 'totalTradedVolume', 'impliedVolatility', 'lastPrice', 'change', 'pChange',
            'totalBuyQuantity', 'totalSellQuantity', 'bidQty', 'bidprice', 'askQty', 'askPrice', 'underlyingValue',
            'type', 'Time', 'delta_t', 'gamma_t', 'theta_t', 'rho_t', 'vega_t', 'price_t', 'impliedVolatility_t']]
        df1 = df1[
            ['strikePrice', 'expiryDate', 'underlying', 'identifier', 'openInterest', 'changeinOpenInterest',
            'pchangeinOpenInterest', 'totalTradedVolume', 'impliedVolatility', 'lastPrice', 'change', 'pChange',
            'totalBuyQuantity', 'totalSellQuantity', 'bidQty', 'bidprice', 'askQty', 'askPrice', 'underlyingValue',
            'type', 'Time', 'delta_t', 'gamma_t', 'theta_t', 'rho_t', 'vega_t', 'price_t', 'impliedVolatility_t']]

    df = pd.concat([df, df1])
    df_list.append(df1.to_dict('records'))
    with open(oi_filename, "w") as files:
        files.write(json.dumps(df_list, indent=4, sort_keys=True))
    print("Written to Jason File1")
    return df

def main():
    dir_write = 0
    global df_list
    try:
        df_list = json.loads(open(oi_filename).read())
    except Exception as error:
        print("Error reading data: {0}".format(error))
        df_list = []

    if df_list:
        df = pd.DataFrame()
        for item in df_list:
            df = pd.concat([df, pd.DataFrame(item)])
    else:
        df = pd.DataFrame()

    time_frame = 3

    while time(9,15) <= datetime.now().time() <= time(15,31):
        timenow = datetime.now()
        #print(timenow.minute)
        check = True if timenow.minute/time_frame in list(np.arange(0.0, 20.0)) else False
        if check:
            nextscan = timenow + timedelta(minutes = time_frame)
            df = fetch_oi(df)
            if not df.empty:
                df['impliedVolatility'] = df['impliedVolatility'].replace(to_replace=0, method='bfill').values
                waitsec = int((nextscan - datetime.now()).seconds)
                if waitsec > time_frame*60:
                    waitsec = time_frame*60
                print("Waiting for {0} seconds".format(waitsec))
                if dir_write == 0:
                    dir_contents()
                    dir_write = 1
                sleep(waitsec) if waitsec > 0 else sleep(0)
            else:
                print("NO data Recieved")
                sleep(30)

if __name__ == '__main__':
    main()
