import numpy as np
import scipy.special


class BankAccount:
    def __init__(self, r):
        self.r = r

    def get_bank_account_value(self, t):
        return (1 + self.r) ** t


class MarketInstrument:
    def __init__(self, S_0, p):
        self.S_0 = S_0
        self.p = p

    def get_instrument_value(self, d, u, t):
        U = np.random.binomial(1, self.p, t)
        return self.S_0 * u ** (sum(U)) * d ** (t - sum(U))


class CRRMarket:
    def __init__(self, S_0, d, u, r, T):
        self.B = BankAccount(r)
        self.p = (1 + r - d) / (u - d)
        self.S = MarketInstrument(S_0, self.p)
        self.d = d
        self.u = u
        self.r = r
        self.T = T


example_market = CRRMarket(100, 0.8, 1.3, 0.1, 10)
result = example_market.S.get_instrument_value(example_market.d, example_market.u, 10)
print(result)


def compute_call_option(t, j, market, K):
    result = market.S.S_0 * market.u ** j * market.d ** (t - j) - K
    return result if result > 0 else 0


def theoretical_value(t, market: CRRMarket, K):
    scaling_value = 1 / market.B.get_bank_account_value(t)
    sum = 0
    for j in range(0, t):
        # S_0 should be S_T-t
        sum += scipy.special.binom(t, j) * market.p ** j * (1 - market.p) ** (t - j) * compute_call_option(t, j, market, K)
    return sum * scaling_value

print(theoretical_value(10, example_market, 90))
