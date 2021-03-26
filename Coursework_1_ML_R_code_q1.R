


###   MATH97131- Machine Learning   ###

# Coursework 1 - Spring 2021 #


rm(list = objects())
par(mfrow = c(1,1))
setwd("C:/Users/robin/Dropbox/Applications/Overleaf/Coursework 1 ML")

library(xtable)
library(ggplot2)
library(tidyverse)

data = read.csv("01945214.csv")
head(data)
dim(data)

# shuffle data

set.seed(42)
NumOfDataPairs = dim(data)[1]
s = sample.int(NumOfDataPairs)
data_shuffled = data[s,]


# creation design matrix

y = data$y
x_1 = data$X.1
x_2 = data$X.2

png(filename = "y_x_1.png")
g = ggplot(data = data)
g = g + geom_point(aes(x = x_1, y = y)) + theme(text = element_text(size=20))
g
graphics.off()

png(filename = "y_x_2.png")
g = ggplot(data = data)
g = g + geom_point(aes(x = x_2, y = y)) + theme(text = element_text(size=20))
g
graphics.off()

X = cbind(1, cos(x_1), cos(x_2))

PolyOrder = 4

for (p in 1:PolyOrder){
  X = cbind(X, x_1^p)
}

for (p in 1:PolyOrder){
  X = cbind(X, x_2^p)
}

# Compute average MSE using a k folds CV procedure for a given lambda.

k_cv_lambda = function(X,y,k=10 ,lambda=0){
  # X is the matrix of features (including cosinus and polynomials)
  # y is a vector containing the targets
  # Lambda is the parameter for ridge regression
  
  NumOfDataPairs = length(y)
  size_test_set = NumOfDataPairs %/% k
  
  # Initialize CV variable for storing results
  CV = matrix(nrow=k, ncol=1)
  
  for (i in 0:(k-1)){
    
    s = seq(i * size_test_set + 1, size_test_set * (i+1))
    
    # Create training design matrix and target data, leaving one out each time
    Train_X = X[-s, ]
    Train_y = y[-s]
    
    # Create testing design matrix and target data
    Test_X = X[s, ]
    Test_y = y[s]
    
    # Learn the optimal parameters using MSE loss
    
    Paras_hat = solve( t(Train_X) %*% Train_X + NumOfDataPairs * lambda*diag(dim(X)[2]) ) %*% t(Train_X) %*% Train_y
    Pred_y = Test_X %*% Paras_hat
    
    # Calculate the MSE of prediction using training data
    CV[i+1] = mean((Pred_y - Test_y)^2) 
    
  }
  return(mean(CV))
}


## Find best parameter lambda

png(filename = "MSE_lambda_dezoom.png")
lambdalist = seq(0.0001,1500,by=0.2)
MSE_lambda = sapply(lambdalist, function(lambda) k_cv_lambda(X,y,k=10,lambda=lambda))
data_lambda = data.frame(lambda = lambdalist, mse = MSE_lambda)
g = ggplot(data_lambda, aes(x = lambda, y = mse)) + geom_line()
g = g + scale_x_log10() + theme(legend.position="right", text = element_text(size=20)) + xlab(expression(lambda)) + ylab("MSE")
g
dev.off()

png(filename = "MSE_lambda_zoom.png")
lambdalist_2 = seq(0,0.02,by=0.0001)
MSE_lambda = sapply(lambdalist_2, function(lambda) k_cv_lambda(X,y,k=10,lambda=lambda))
data_lambda = data.frame(lambda = lambdalist_2, mse = MSE_lambda)
lambda <-lambdalist_2[which.min(MSE_lambda)]
g = ggplot(data_lambda, aes(x = lambda, y = mse)) + geom_line()
g = g + theme(legend.position="right", text = element_text(size=20)) + xlab(expression(lambda)) + ylab("MSE")
g = g + geom_vline(xintercept = lambda, color = 'red', lty = 2)
g = g + annotate(geom="text", x=0.006, y=4.2025, label= "lambda_hat = 0.025",
         color="red")
g
dev.off()

# predictions for new observations

Paras_hat_lambda = solve( t(X) %*% X + NumOfDataPairs * lambda * diag(dim(X)[2]) ) %*% t(X) %*% y
xtable(round(t(Paras_hat_lambda),digits = 4), digits = 4)

X1 = c(-2.1, 2.9, -0.9, 3.8)
X2 = c(4.4, -4.5, 0.3, 3.9)

new_X = cbind(1, cos(X1), cos(X2))

PolyOrder = 4

for (p in 1:PolyOrder){
  new_X = cbind(new_X, X1^p)
}

for (p in 1:PolyOrder){
  new_X = cbind(new_X, X2^p)
}

Pred_y = new_X %*% Paras_hat_lambda
xtable(round(t(Pred_y), digits = 4))


# 2

sigma = 2
posterior_mean = function(alpha,X,y){
  gram = t(X) %*% X
  d = dim(gram)[1]
  I = diag(rep(1,d))
  return (solve(gram + (sigma^2/alpha) * I) %*% t(X) %*% y)
}

posterior_mean(alpha = 2,X,y)

png(filename = "plot_post_mean.png", width = 650)

list_alpha = seq(0.0000001,1000,by=0.1)

post_means = sapply(1:11, function(i) sapply(list_alpha, function(alpha) posterior_mean(alpha = alpha,X,y)[i]))

dat = data.frame(alpha = list_alpha, post_means)
colnames(dat) = c("alpha", paste("theta",letters[1:11]))
dat = dat %>%  gather(key = "estimates", value = "value", -alpha)

g = ggplot(dat, aes(x = alpha, y = value)) + geom_line(aes(color = estimates))
g = g + scale_x_log10() + theme(legend.position="right", text = element_text(size=15)) + xlab(expression(alpha)) + ylab("posterior mean")
g = g + scale_color_discrete(labels = paste("theta",0:10))
g

dev.off()

posterior_mean(alpha=1,X,y)

alpha_ult = sigma^2/(lambda * NumOfDataPairs)

table = data.frame(
Paras_hat_lambda,
posterior_mean(alpha=0.0000000001,X,y),
posterior_mean(alpha=1,X,y),
posterior_mean(alpha=alpha_ult,X,y),
posterior_mean(alpha=100000000,X,y)
)

xtable(t(table), digits = 4)

mylm = lm(y~X-1)
mylm$coefficients

# predictions alpha = 1

alpha = 1
mu = solve(t(X) %*% X + (sigma^2/alpha) * NumOfDataPairs * diag(dim(X)[2])) %*% t(X) %*% y
Eps = sigma^2 * solve(t(X) %*% X + (sigma^2/alpha) * diag(dim(X)[2]))

mean_pred = new_X %*% mu
sd_pred = new_X %*% Eps %*% t(new_X) + sigma^2 * diag(4)

round(mean_pred, digits = 4)
round(sd_pred, digits = 4)


