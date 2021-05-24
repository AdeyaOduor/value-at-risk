# value-at-risk
Value at risk (VaR) is a statistic used to try and quantify the level of financial risk within a firm or portfolio over a specified time frame. VaR provides an estimate of the maximum loss from a given position or portfolio over a period of time, and you can calculate it across various confidence levels.
Before we get started, note that the standard VaR calculation assumes the following:

    Normal distribution of returns - VaR assumes the returns of the portfolio are normally distributed. This is of course not realistic for most assets, but allows us to develop a baseline using a much more simplistic calculation.
        (Modifications can be made to VaR to account for different distributions, but here we'll focus on the standard VaR calculation)

    Standard market conditions - Like many financial instruments, VaR is best used for considering loss in standard markets, and is not well-suited for extreme/outlier events.
In order to calculate the VaR of a portfolio, you can follow the steps below:

    Calculate periodic returns of the stocks in the portfolio
    Create a covariance matrix based on the returns
    Calculate the portfolio mean and standard deviation
        (weighted based on investment levels of each stock in portfolio)
    Calculate the inverse of the normal cumulative distribution (PPF) with a specified confidence interval, standard deviation, and mean
    Estimate the value at risk (VaR) for the portfolio by subtracting the initial investment from the calculation in step (4)
