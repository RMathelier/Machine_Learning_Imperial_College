


###   MATH97131- Machine Learning   ###

# Coursework 1 - Spring 2021 #

rm(list = objects())
setwd("C:/Users/robin/Dropbox/Applications/Overleaf/Coursework 1 ML")

library(xtable)
library(ggplot2)
library(tidyverse)

# 1

list_eps = c(0.1, 1, 2)
list_p = seq(1,15,by=1)
ord_eps_1 = 1/(2*(1+list_eps[1])^list_p)

png(filename = "Bayes_error_rate.png")

data_eps_p = sapply(1:length(list_eps), function(i) 1/(2*(1+list_eps[i])^list_p))
data_eps_p = data.frame(cbind(list_p, data_eps_p))
colnames(data_eps_p) = c('p',letters[1:3])
data_eps_p = data_eps_p %>%  gather(key = "epsilon", value = "value", -p)

g = ggplot(data_eps_p, aes(x = p, y = value)) + geom_line(aes(color = epsilon))
g = g + theme(legend.position="right", text = element_text(size=20)) + xlab("p") + ylab("Bayes error rate")
g = g + scale_color_discrete(labels = c(0.1,1,2))
g = g + geom_point(aes(x = p, y = value))
g

dev.off()


# 2

set.seed(01945214)

n = 1000
p = 4
eps = 0.5

x = matrix(nrow = n,ncol = p)
y = c()

for (i in 1:n){
  u = runif(1)
  if (u < 0.5){
    y = c(y,1)
    x[i,] = runif(p)
  }
  if (u > 0.5){
    y = c(y,2)
    x[i,] = runif(p, min = 0, max = 1 + eps)
  }
}

data = data.frame(x,y = as.factor(y))
head(data)
dim(data)


##   Part a   ##

library(caret)
library(tictoc)

# training logistic regression

tic()

train_control_log_reg = trainControl(method = "cv", number = 10)
log_reg = train(y ~ .,
               data = data,
               trControl = train_control_log_reg,
               method = "glm",
               family=binomial())
toc()

round(1-log_reg$results$Accuracy, digits = 4) * 100
round(log_reg$results$AccuracySD, digits = 4) * 100

# training qda

tic()
train_control_qda = trainControl(method = "cv", number = 10)
qda_clf = train(y ~ .,
                data = data,
                trControl = train_control_qda,
                method = "qda")
toc()

round(1-qda_clf$results$Accuracy, digits = 4) * 100
round(qda_clf$results$AccuracySD, digits = 4) * 100

##   Part b   ##

# function to create the train/test sets for outer CV

create_outer_CV_sets = function(X, y, n_folds = 10){
  
  # create subsets for outer cross-validation
  
  NumOfDataPairs = length(y)
  size_test_set = NumOfDataPairs %/% n_folds
  
  outer_CV_train_sets = list()
  outer_CV_test_sets = list()
  outer_CV_train_sets_label = list()
  outer_CV_test_sets_label = list()
  
  for (i in 0:(n_folds-1)){
    
    s = seq(i * size_test_set + 1, size_test_set * (i+1))
    
    # Create training design matrix and target data
    X_train = X[-s, ]
    y_train = y[-s]
    
    # Create testing design matrix and target data
    X_test = X[s, ]
    y_test = y[s]
    
    outer_CV_train_sets[[i+1]] = X_train
    outer_CV_train_sets_label[[i+1]] = y_train
    
    outer_CV_test_sets[[i+1]] = X_test
    outer_CV_test_sets_label[[i+1]] = y_test
    
  }
  return(list(outer_CV_train_sets, outer_CV_train_sets_label, outer_CV_test_sets, outer_CV_test_sets_label))
}


# Nested CV for knn

k_nn_cv <- function(x, y, n_folds_outer = 10, n_folds_inner = 5, list_k = 1:20){
  # X is the matrix of features 
  # y is a vector containing the targets
  
  outer_CV_train_sets = create_outer_CV_sets(x, y, n_folds = n_folds_outer)[[1]]
  outer_CV_train_sets_label = create_outer_CV_sets(x, y, n_folds = n_folds_outer)[[2]]
  
  outer_CV_test_sets = create_outer_CV_sets(x, y, n_folds_outer)[[3]]
  outer_CV_test_sets_label = create_outer_CV_sets(x, y, n_folds_outer)[[4]]
  
  cv_k = c()

  for (k in list_k){
    
    cv_errors_k = c()
    
    for (i in 1:length(outer_CV_train_sets)){
      
      inner_CV_train_set = outer_CV_train_sets[[i]]
      inner_CV_train_set_label = outer_CV_train_sets_label[[i]]
      inner_CV_test_set = outer_CV_test_sets_label[[i]]
      
      inner_data = data.frame(x_inner = inner_CV_train_set, y_inner = as.factor(inner_CV_train_set_label))
      
      trControl = trainControl(method  = "cv",
                                number  = n_folds_inner)
      
      fit_knn_k = train(y_inner ~ .,
                   method     = "knn",
                   tuneGrid = data.frame(k = k),
                   trControl  = trControl,
                   metric     = "Accuracy",
                   data       = inner_data)
      
      cv_errors_k = c(cv_errors_k, fit_knn_k$results$Accuracy)

    }
    
    cv_k = c(cv_k, mean(cv_errors_k))
  }
  
  best_k = list_k[which.max(cv_k)]
  
  outer_data = data.frame(x_outer = x, y_outer = as.factor(y))
  
  trControl = trainControl(method  = "cv",
                            number  = n_folds_outer)
  
  fit_knn_best_k = train(y_outer ~ .,
               method     = "knn",
               tuneGrid = data.frame(k = best_k),
               trControl  = trControl,
               metric     = "Accuracy",
               data       = outer_data)
  
  return(list(fit_knn_best_k,best_k))
}

tic()
fitted_best_knn = k_nn_cv(x, y)
toc()

knn_clf = fitted_best_knn[[1]]
best_k = fitted_best_knn[[2]]

round(1-knn_clf$results$Accuracy, digits = 4)*100
round(knn_clf$results$AccuracySD, digits = 4)*100

# d

bayes_error = 1/(2*(1+eps)^p)
round(bayes_error, digits = 4)

# lowest possible test error rate for the classifier

errors_bayes_clf = c()

K = 10
size_test_set = n %/% K

tic()
for (i in 0:(K-1)){
  s = seq(i * size_test_set + 1, size_test_set * (i+1))
  y_pred = c()
  sapply(s, function(i) {
    if (sum(x[i,] > 1) >= 1) y_pred <<- c(y_pred, 2)
    else y_pred <<- c(y_pred, 1)
    }
    )
  errors_bayes_clf[i+1] = mean((y_pred - y[s])^2)

}
toc()

round(mean(errors_bayes_clf), digits = 4) * 100
round(sd(errors_bayes_clf), digits = 4) * 100


