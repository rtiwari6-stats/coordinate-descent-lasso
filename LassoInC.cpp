#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// Soft-thresholding function, returns scalar
// [[Rcpp::export]]
double soft_c(double a, double lambda){
  // Your function code goes here
  if(a > lambda){
    return (a-lambda);
  }
  if(a < -lambda){
    return (a + lambda);
  }
  else{
    return 0.0;
  }
}

// Lasso objective function, returns scalar
// [[Rcpp::export]]
double lasso_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& beta, double lambda){
  // Your function code goes here
  int n = Xtilde.n_rows;
  
  return  accu(square(Ytilde - Xtilde * beta)/(2 * n)) + lambda * accu(abs(beta));;
}

// Lasso coordinate-descent on standardized data with one lamdba. Returns a vector beta.
// [[Rcpp::export]]
arma::colvec fitLASSOstandardized_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, double lambda, const arma::colvec& beta_start, double eps = 0.001){
  // Your function code goes here
  int n = Xtilde.n_rows;
  int p = Xtilde.n_cols;
  
  double error = 10;
  arma::colvec beta_old = beta_start;
  arma::colvec beta_new = beta_start;
  arma::colvec residual = Ytilde - Xtilde * beta_start;
  while(error >= eps){
    beta_old = beta_new;
    double obj_old = lasso_c(Xtilde, Ytilde, beta_old, lambda);
    for(int i = 0; i < p; i++)
    {
      // Update beta
      beta_new(i) = soft_c(arma::as_scalar(beta_old(i) + (Xtilde.col(i).t() * residual / n)), lambda);
      // Update partial residual
      residual = residual + Xtilde.col(i) * (beta_old(i) - beta_new(i));
    }
    double obj_new = lasso_c(Xtilde, Ytilde, beta_new, lambda);
    error = obj_old - obj_new;
  }
  return beta_new;
}  

// Lasso coordinate-descent on standardized data with supplied lambda_seq. 
// You can assume that the supplied lambda_seq is already sorted from largest to smallest, and has no negative values.
// Returns a matrix beta (p by number of lambdas in the sequence)
// [[Rcpp::export]]
arma::mat fitLASSOstandardized_seq_c(const arma::mat& Xtilde, const arma::colvec& Ytilde, const arma::colvec& lambda_seq, double eps = 0.001){
  // Your function code goes here
  size_t p = Xtilde.n_cols;
  
  arma::mat beta_mat(p, lambda_seq.size());
  arma::colvec beta_start(p);
  beta_start.fill(0.0); //warm start
  for(size_t i = 0; i < lambda_seq.size(); i++)
  {
    arma::colvec beta = fitLASSOstandardized_c(Xtilde,  Ytilde, lambda_seq(i), beta_start, eps);
    beta_mat.col(i) = beta;
    beta_start = beta; // warm start
  }
  
  return beta_mat; 
}