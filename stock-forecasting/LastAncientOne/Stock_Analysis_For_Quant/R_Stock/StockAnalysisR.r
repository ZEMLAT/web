options(warn=-1) #Ignore Warning

library(quantmod)
library(PerformanceAnalytics)

SPY = getSymbols("SPY", from="2001-01-01", auto.assign=F)

head(SPY)

tail(SPY)

summary(SPY)

plot(SPY)

# Calculate Returns
rets = ROC(Cl(SPY), type="discrete")

# Another way to calculate returns
returns <- Return.calculate(Cl(SPY))
head(returns)

head(rets)

plot(rets, main="")
title(main='Stock Returns', cex=1.5, font=4)

table.Drawdowns(rets["2008/"], top=10)


