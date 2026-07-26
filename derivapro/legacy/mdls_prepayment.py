import QuantLib as ql
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

class Prepayment:
    def __init__(self, orig_rate, market_rate, orig_fico, loan_age, orig_ltv, intercept, beta_spread, beta_fico, beta_loan_age, beta_ltv):
        self.orig_rate = orig_rate
        self.market_rate = market_rate
        self.orig_fico = orig_fico
        self.loan_age = loan_age
        self.orig_ltv = orig_ltv
        self.intercept = intercept
        self.beta_spread = beta_spread
        self.beta_fico = beta_fico
        self.beta_loan_age = beta_loan_age
        self.beta_ltv = beta_ltv
    

    def prepayment_probability(self):
        refi_incentive = self.orig_rate - self.market_rate

        log_odds = (
            self.intercept +
            self.beta_spread * refi_incentive +
            self.beta_fico * (self.orig_fico - 700) +  # center FICO
            self.beta_loan_age * self.loan_age +
            self.beta_ltv * self.orig_ltv
            )
        probability = round(1 / (1 + np.exp(-log_odds)) * 100, 2)
        return probability


# print("Prepayment Probability: ", odds.prepayment_probability(), "%")



