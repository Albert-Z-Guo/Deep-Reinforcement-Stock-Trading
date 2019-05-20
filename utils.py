import numpy as np


def format_price(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def stock_close_prices(key):
	'''return a list containing stock close prices from a .csv file'''
	prices = []
	lines = open("data/" + key + ".csv", "r").read().splitlines()
	for line in lines[1:]:
		prices.append(float(line.split(",")[4]))
	return prices


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def generate_state(stock_prices, t, n):
	'''
	return an n-day state representation ending at time t
	the state is defined as the adjacent daily stock price differences for a n-day period

	'''
	d = t - n + 1
	block = stock_prices[d:t + 1] if d >= 0 else -d * [stock_prices[0]] + stock_prices[0:t + 1] # pad with t_0
	res = []
	for i in range(n - 1):
		res.append(sigmoid(block[i + 1] - block[i]))
	return np.array([res])
