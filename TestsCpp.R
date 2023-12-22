
# Header for Rcpp and RcppArmadillo
library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
sourceCpp("LassoInC.cpp")

# Source your LASSO functions from HW4 (make sure to move the corresponding .R file in the current project folder)
source("LassoFunctions.R")

# Do at least 2 tests for soft-thresholding function below. You are checking output agreements on at least 2 separate inputs
#################################################

#test when a is greater than lambda
test_a_greater_lambda = function(){
  lambda=6
  a = 10
  stopifnot((a-lambda) == soft(a,lambda))
  stopifnot(soft_c(a,lambda) == soft(a,lambda))
}
#test when |a| <= lambda
test_mod_a_less_equal_lambda = function(){
  lambda=6
  a = -1
  stopifnot(0== soft(a,lambda))
  stopifnot(soft_c(a,lambda) == soft(a,lambda))
  
  lambda=5.5
  a = -0.75
  stopifnot(0== soft(a,lambda))
  stopifnot(soft_c(a,lambda) == soft(a,lambda))
}

#test when a < -lambda
test_a_less_minus_lambda = function(){
  lambda= -6
  a = 1
  stopifnot(7== soft(a,lambda))
  stopifnot(soft_c(a,lambda) == soft(a,lambda))
  
  lambda= 5.5
  a = -7.75
  stopifnot(-2.25== soft(a,lambda))
  stopifnot(soft_c(a,lambda) == soft(a,lambda))
}
#Run tests
test_a_greater_lambda()
test_mod_a_less_equal_lambda()
test_a_less_minus_lambda()

# Do at least 2 tests for lasso objective function below. You are checking output agreements on at least 2 separate inputs
#################################################
#test with small parameters
test_small_parameters = function(){
  set.seed(100)
  p = 5
  n = 50
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  0.01
  beta =  rep(0, p) #zero beta
  
  rout = lasso(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  cout = lasso_c(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  stopifnot(all.equal(as.numeric(rout), cout))
  
  set.seed(100)
  p = 5
  n = 50
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  0.01
  beta =  rnorm(p) #random beta
  
  rout = lasso(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  cout = lasso_c(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  stopifnot(all.equal(as.numeric(rout), cout))
}

#test with large parameters
test_large_parameters = function(){
  set.seed(100)
  p = 50
  n = 5000
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  0.05
  beta =  rep(0, p) #zero beta
  
  rout = lasso(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  cout = lasso_c(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  stopifnot(all.equal(as.numeric(rout), cout))
  
  set.seed(100)
  p = 50
  n = 5000
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  0.5
  beta =  rnorm(p) #random beta
  
  rout = lasso(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  cout = lasso_c(standardized$Xtilde, standardized$Ytilde, beta, lambda)
  stopifnot(all.equal(as.numeric(rout), cout))
}

#run tests
test_small_parameters()
test_large_parameters()

# Do at least 2 tests for fitLASSOstandardized function below. You are checking output agreements on at least 2 separate inputs
#################################################
test_fitLassoStandardized = function(){
  set.seed(100)
  p = 50
  n = 5000
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  0
  beta =  rnorm(p, 5, 1) #random beta
  rout = fitLASSOstandardized(standardized$Xtilde, standardized$Ytilde, 
                              lambda = lambda, beta_start=beta, eps = 1e-5)
  cout = fitLASSOstandardized_c(standardized$Xtilde, standardized$Ytilde, 
                              lambda = lambda, beta_start=beta, eps = 1e-5)
  stopifnot(all.equal(rout$beta, as.numeric(cout)))
  
  set.seed(100)
  p = 5
  n = 100
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized <- standardizeXY(X, Y)
  lambda =  1e-4
  beta =  rnorm(p, 5, 1) #random beta
  rout = fitLASSOstandardized(standardized$Xtilde, standardized$Ytilde, 
                              lambda = lambda, beta_start=beta, eps = 1e-5)
  cout = fitLASSOstandardized_c(standardized$Xtilde, standardized$Ytilde, 
                                lambda = lambda, beta_start=beta, eps = 1e-5)
  stopifnot(all.equal(rout$beta, as.numeric(cout)))
}
test_fitLassoStandardized()

# Do microbenchmark on fitLASSOstandardized vs fitLASSOstandardized_c
######################################################################
set.seed(100)
p = 100
n = 10000
Y = rnorm(n)
X =  matrix(rnorm(n * p), n, p)
standardized <- standardizeXY(X, Y)
lambda =  1e-4
beta_start =  rnorm(p, 5, 1) #random beta
library(microbenchmark)
microbenchmark(
  fitLASSOstandardized(standardized$Xtilde, standardized$Ytilde, lambda = lambda, 
                       beta_start = beta_start, eps = 1e-5),
  fitLASSOstandardized_c(standardized$Xtilde, standardized$Ytilde, lambda = lambda, beta_start = beta_start, eps = 1e-5),
  times = 10
)
#cpp median time is 15.30195 milliseconds. R median time is 94.01505 milliseconds.

# Do at least 2 tests for fitLASSOstandardized_seq function below. You are checking output agreements on at least 2 separate inputs
#################################################
test_fitLASSOstandardized_seq = function(){
  set.seed(100)
  p = 5
  n = 100
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized = standardizeXY(X, Y)
  lambda_max = max(abs(crossprod(standardized$Xtilde, standardized$Ytilde))) / n
  lambda_seq = exp(seq(log(lambda_max), log(0.01), length = 10))
  rout = fitLASSOstandardized_seq(standardized$Xtilde, standardized$Ytilde, 
                                  lambda_seq = lambda_seq, eps = 1e-5)
  cout = fitLASSOstandardized_seq_c(standardized$Xtilde, standardized$Ytilde, 
                                    lambda_seq = lambda_seq, eps = 1e-5)
  stopifnot(all.equal(rout$beta, (cout)))
  
  set.seed(100)
  p = 50
  n = 1000
  Y = rnorm(n)
  X =  matrix(rnorm(n * p), n, p)
  standardized = standardizeXY(X, Y)
  lambda_max = max(abs(crossprod(standardized$Xtilde, standardized$Ytilde))) / n
  lambda_seq = exp(seq(log(lambda_max), log(0.01), length = 5))
  rout = fitLASSOstandardized_seq(standardized$Xtilde, standardized$Ytilde, 
                                  lambda_seq = lambda_seq, eps = 1e-6)
  cout = fitLASSOstandardized_seq_c(standardized$Xtilde, standardized$Ytilde, 
                                    lambda_seq = lambda_seq, eps = 1e-6)
  stopifnot(all.equal(rout$beta, (cout)))
}
test_fitLASSOstandardized_seq()


# Do microbenchmark on fitLASSOstandardized_seq vs fitLASSOstandardized_seq_c
######################################################################
set.seed(100)
p = 50
n = 1000
Y = rnorm(n)
X =  matrix(rnorm(n * p), n, p)
standardized = standardizeXY(X, Y)
lambda_max = max(abs(crossprod(standardized$Xtilde, standardized$Ytilde))) / n
lambda_seq = exp(seq(log(lambda_max), log(0.01), length = 10))
library(microbenchmark)
microbenchmark(fitLASSOstandardized_seq(standardized$Xtilde, standardized$Ytilde, 
                                        lambda_seq = lambda_seq, eps = 1e-6),
               fitLASSOstandardized_seq_c(standardized$Xtilde, standardized$Ytilde, 
                                          lambda_seq = lambda_seq, eps = 1e-6),
               times=10)
#median time for cpp is 2.62045 milliseconds and for R is 19.97865 milliseconds.

# Tests on riboflavin data
##########################
require(hdi) # this should install hdi package if you don't have it already; otherwise library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene erpression

# Make sure riboflavin$x is treated as matrix later in the code for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]

# Standardize the data
out <- standardizeXY(riboflavin$x, riboflavin$y)

# This is just to create lambda_seq, can be done faster, but this is simpler
outl <- fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, n_lambda = 30)

# The code below should assess your speed improvement on riboflavin data
microbenchmark(
  fitLASSOstandardized_seq(out$Xtilde, out$Ytilde, outl$lambda_seq),
  fitLASSOstandardized_seq_c(out$Xtilde, out$Ytilde, outl$lambda_seq),
  times = 10
)
#median time on R is 1048.0894 milliseonds and on cpp is 29.4854 milliseonds.