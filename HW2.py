import numpy as np
import yfinance as yf
from scipy.stats import *
from sklearn.linear_model import LinearRegression


tickers = ["AAPL", "MSFT", "SBUX", "MCD", "TSLA"]
start_date = "2018-02-19"
end_date = "2023-02-20"
data = yf.download(tickers, start = start_date, end = end_date, interval = "1d")["Adj Close"]
r = data.pct_change().dropna()

AAPL_r = r["AAPL"]
MSFT_r = r["MSFT"]
SBUX_r = r["SBUX"]
MCD_r = r["MCD"]
TSLA_r = r["TSLA"]

std = r.std()
M = r.corr()
portfolio_size = 100000000
CI_level = 0.95


# 1 Full covariance model
P = np.zeros(len(std))      # 建立一個與投資組合中資產數量相同長度的零陣列 P
for i in range(len(std)):       
    P[i] = 0.2 * portfolio_size * std.iloc[i]
    
portfolio_std = np.sqrt(np.dot(P.T, np.dot(M, P)))      # σₚ = √(PᵀMP)
n = norm.ppf(1 - CI_level)
VaR_full = abs(n * portfolio_std)       
print(f"Full covariance model VaR (1-day, 95%) : {VaR_full:.2f}")


# 2 Diagonal model
sp500 = yf.download("^GSPC", start = start_date, end = end_date, interval = "1d")["Adj Close"]
sp500_r = sp500.pct_change().dropna()
model = LinearRegression()

# 將一維 series 轉為二維數組
sp_r1 = sp500_r.values.reshape(-1, 1)
aa_r1 = AAPL_r.values.reshape(-1, 1)
ms_r1 = MSFT_r.values.reshape(-1, 1)
sb_r1 = SBUX_r.values.reshape(-1, 1)
mc_r1 = MCD_r.values.reshape(-1, 1)
ts_r1 = TSLA_r.values.reshape(-1, 1)

def fit_model(x, y):
    model.fit(x, y)
    beta = model.coef_.flatten()    # model.coef_ 返回的是一個二維數組，但是只需要一維的係數，因此使用 flatten 將其壓縮為一維。
    error = y - model.predict(x)
    error_var = np.var(error)       # 誤差的平均平方，即 MSE
    return beta, error_var

# Fit models
betas = {}
error_vars = {}
for stock, data in zip(['aa', 'ms', 'sb', 'mc', 'ts'], [aa_r1, ms_r1, sb_r1, mc_r1, ts_r1]):        # zip() 函式將資料集的名稱與對應的資料一一配對
    betas[stock], error_vars[stock] = fit_model(sp_r1, data)

W = np.ones(5) * 0.2 * portfolio_size
B = np.array([betas[stock] for stock in ['aa', 'ms', 'sb', 'mc', 'ts']])      # 二維 NumPy 數組
error_var = [error_vars[stock] for stock in ['aa', 'ms', 'sb', 'mc', 'ts']]       # 包含5個元素的列表
error_mat = np.diag(error_var)

VRp = np.sqrt(np.dot(W.T, np.dot(B, np.dot(B.T, W))) * np.var(sp500_r) + np.dot(W.T, np.dot(error_mat, W)))     # V(Rp) = (Wᵀββᵀw) σₘ² + wᵀDε w
VaR_diag = abs(n * VRp)     # n 表 confidence level
print(f"Diagonal model VaR (1-day, 95%) : {VaR_diag:.2f}")


# 3 Beta model
VRp = np.sqrt(np.dot(W.T, np.dot(B, np.dot(B.T, W))) * np.var(sp500_r))     # V(Rp) = (Wᵀββᵀw) σₘ²
VaR_beta = abs(n * VRp)
print(f"Beta model VaR (1-day, 95%) : {VaR_beta:.2f}")


# 4 Undiversified model
n_w = n * 0.2 * portfolio_size
VaR_undiv = 0
for i in range(len(std)):
    VaR_undiv += abs(n_w * std.iloc[i])

print(f"Undiversified model VaR (1-day, 95%) : {VaR_undiv:.2f}")
