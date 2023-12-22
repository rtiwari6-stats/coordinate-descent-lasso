# [ToDo] Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  n = nrow(X)
  p = ncol(X)
  if(n != length(Y)){
    stop("Length of vector Y should be equal to the number of rows of matrix X")
  }
  # [ToDo] Center Y
  Ymean = mean(Y)
  Ytilde = Y - Ymean
  
  # [ToDo] Center and scale X
  Xmeans = colMeans(X)
  Xscaled = t(t(X) - Xmeans)
  weights = sqrt(colSums(Xscaled ^ 2)/n)
  Xtilde = Xscaled %*% diag(1/weights)
  
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# [ToDo] Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  return(sign(a)*max(abs(a)-lambda,0))
}

# [ToDo] Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  #xtilde %*% beta is nx1
  return((crossprod(Ytilde - Xtilde %*% beta)) /(2 * length(Ytilde)) 
         + lambda * sum(abs(beta)))
}

# [ToDo] Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  #[ToDo]  Check that n is the same between Xtilde and Ytilde
  n = nrow(Xtilde)
  p = ncol(Xtilde)
  if(n != length(Ytilde)){
    stop("Number of rows of Xtilde must be equal to length of Ytilde")
  }
  #[ToDo]  Check that lambda is non-negative
  if(lambda < 0){
    stop("Lambda must be non-negative")
  }
  
  #[ToDo]  Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  if(is.null(beta_start)) {
    beta_start = as.vector(rep(0,p))
  } 
  else {
    if(p != length(beta_start)) { 
      stop("Length of beta_start must be equal to the number of columns in Xtilde") 
    }
  }
  
  #[ToDo]  Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  error <- 10
  beta_old = beta_start
  beta_new = beta_start
  residual = Ytilde - Xtilde %*% beta_start
  while(error >= eps){
    beta_old = beta_new
    for(i in 1:p){
      beta_new[i] = soft((beta_old[i] + 
                            crossprod(residual, Xtilde[, i])/n), lambda)
      residual = residual + Xtilde[, i] * (beta_old[i] - beta_new[i])
    }
    error = lasso(Xtilde, Ytilde, beta_old, lambda) - lasso(Xtilde, Ytilde, beta_new, lambda)
  }
  fmin = lasso(Xtilde, Ytilde, beta_new, lambda)
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta_new, fmin = fmin))
}

# [ToDo] Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Check that n is the same between Xtilde and Ytilde
  n = nrow(Xtilde)
  p = ncol(Xtilde)
  if(n != length(Ytilde)){
    stop("Number of rows of Xtilde must be equal to length of Ytilde")
  }
  # [ToDo] Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  if(!is.null(lambda_seq)) {
    lambda_seq = lambda_seq[lambda_seq >= 0]
    if(length(lambda_seq) > 0) {
      lambda_seq = sort(lambda_seq[lambda_seq >= 0], decreasing = TRUE)
    }
    else{
      lambda_seq = NULL
      writeLines("Warning: None of the supplied values in lambda_seq are >=0. Generating values as if they were not supplied.")
    }
  }
  
  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution),
  # and create a sequence of length n_lambda as
  if(is.null(lambda_seq)){
    lambda_max = max(abs(crossprod(Xtilde, Ytilde))) / n
    lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  }
  
  # [ToDo] Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  beta_mat =  matrix(0, p, length(lambda_seq))
  fmin_vec = rep(0, length(lambda_seq))
  beta_start = rep(0, p) # warm start strategy
  for (i in 1:length(lambda_seq)) {
    fit = fitLASSOstandardized(Xtilde, Ytilde, lambda = lambda_seq[i], 
                               beta_start = beta_start, eps = eps)
    beta_mat[, i] = fit$beta
    fmin_vec[i] = fit$fmin
    beta_start = fit$beta #for warm start
  }
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# [ToDo] Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Center and standardize X,Y based on standardizeXY function
  standardized = standardizeXY(X, Y)
  Xtilde = standardized$Xtilde
  Ytilde = standardized$Ytilde
  Ymean = standardized$Ymean
  Xmeans = standardized$Xmeans
  weights = standardized$weights
  # [ToDo] Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
  lasso_fit = fitLASSOstandardized_seq(Xtilde, Ytilde, lambda_seq, n_lambda, eps)
 
  # [ToDo] Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  beta_mat = diag(1/weights) %*% lasso_fit$beta_mat
  beta0_vec = Ymean - Xmeans %*% beta_mat
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lasso_fit$lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# [ToDo] Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  # [ToDo] Fit Lasso on original data using fitLASSO
  lassofit = fitLASSO(X ,Y, lambda_seq, n_lambda, eps)
  lambda_seq = lassofit$lambda_seq
  beta_mat = lassofit$beta_mat
  beta0_vec = lassofit$beta0_vec
  n = nrow(X)
  p = ncol(X)
  if(length(Y) != n){
    stop("Length of Y must be equal to the number of rows in X")
  }
 
  # [ToDo] If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  if(is.null(fold_ids) || length(fold_ids) == 0){
    fold_ids = sample(1:n) %% k + 1
  }
  
  # [ToDo] Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
  errors_all = matrix(NA, length(unique(fold_ids)), length(lambda_seq)) # fold-specific information
  for (fold in 1:k){
    #Create training data xtrain and ytrain, everything except fold
    xtrain = X[fold_ids != fold, ]
    ytrain = Y[fold_ids != fold]
    
    #Create testing data xtest and ytest, everything in fold
    xtest = X[fold_ids == fold, ]
    ytest = Y[fold_ids == fold]
    
    #fit lasso on the folds
    fit = fitLASSO(xtrain, ytrain, lambda_seq, n_lambda, eps)
    beta = fit$beta_mat
    beta0 = fit$beta0_vec
    lambda = fit$lambda_seq
    
    # ytest is fold_len x 1
    # result_matrix is fold_len x length(lambda) so we have ytest vector replicated for each lambda
    result_matrix =  matrix(rep(ytest, each = length(lambda)), 
                           length(ytest), byrow = TRUE)
    #beta - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
    #xtest fold_len x p so xbeta is fold_len x length(lambda_seq)
    xbeta = xtest %*% beta
    xintercept = sweep(result_matrix, 2, beta0) #this subtracts the intercept
    errors_all[fold, ] = colMeans((xintercept - xbeta) ^ 2) #for each 1:k
  }
  
  cvm = colMeans(errors_all)
  
  # [ToDo] Find lambda_min
  lambda_min = lambda_seq[which.min(cvm)] #find lambda with minimum cvm

  # [ToDo] Find lambda_1SE
  cvse = apply(errors_all, 2, function(x) sd(x)/sqrt(length(unique(fold_ids))))
  upper_limit = cvm[which.min(cvm)] + cvse[which.min(cvm)]
  lambda_1se = lambda_seq[which.max(cvm <= upper_limit)]
  
  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec, fold_ids = fold_ids, lambda_min = lambda_min, lambda_1se = lambda_1se, cvm = cvm, cvse = cvse))
}

