# ============================================================
# zip.R
# initial : k-means, k-medoids, heirarchical
# ============================================================

zip_em <- function(X, Z, Y, max_iter=200, tol=1e-6, irls_max_iter=50, irls_tol=1e-6) {
  
  X <- as.matrix(X) # explanatory variables in poisson
  Z <- as.matrix(Z) # explanatory variables in logistic(zero part)
  Y <- as.numeric(Y) # response variable
  N <- length(Y) # number of samples
  
  # ------------------------------------------------------------
  # initialization
  # ------------------------------------------------------------
  # beta(poisson)
  beta <- coef(glm(Y ~ X - 1, family = poisson)) # fitting glm(poisson)
  
  # gamma(logistic)
  gamma <- coef(glm(as.numeric(Y == 0) ~ Z - 1, family = binomial)) #fitting glm(logistic)
  
  ll_old <- -Inf
  loglik_history <- numeric()
  iter <- 1
  
  # ------------------------------------------------------------
  # em algirithm
  # ------------------------------------------------------------
  while(iter <= max_iter) {
    
    # 초기 예측값
    mu <- exp(X %*% beta)
    pi_prob <- 1 / (1 + exp(-(Z %*% gamma)))
    
    # ------------------- E-step -------------------
    z_hat_1 <- numeric(N) # expectation of Inflation part 
    z_hat_2 <- rep(1,N) # expectation of count part(y>0 -> 1)
    
    zero_idx <- (Y == 0)
    pi_z <- pi_prob[zero_idx]
    mu_z <- mu[zero_idx]
    
    # Y=0 <- bayes theorem
    z_hat_2[zero_idx] <- ((1-pi_z)*exp(-mu_z)) / (pi_z + (1-pi_z)*exp(-mu_z))
    
    # Q-ftn
    ll_new <- sum(z_hat_1 * log(pi_prob + 1e-15) +
                  z_hat_2 * log(1-pi_prob + 1e-15) +
                  z_hat_2 * (Y*log(mu +1e-15) - mu))
    
    
    loglik_history <- c(loglik_history, ll_new)
    
    if(abs(ll_new - ll_old) < tol) break
    ll_old <- ll_new
    
    # ------------------- M-step -------------------
    # gamma
    step_gamma <- 1
    while (step_gamma <= irls_max_iter) {
      pi_curr <- 1 / (1 + exp(-(Z %*% gamma)))
      
      # W_gamma
      W_gamma <- as.vector(pi_curr * (1 - pi_curr))
      W_gamma <- pmax(W_gamma, 1e-10)
      
      # v_gamma
      v_gamma <- Z %*% gamma + (z_hat_1 - pi_curr) / W_gamma
      
      # gamma update
      gamma_new <- solve(t(Z) %*% (W_gamma * Z)) %*% t(Z) %*% (W_gamma * v_gamma)
      
      if (max(abs(gamma_new - gamma)) < irls_tol) {
        gamma <- as.vector(gamma_new)
        break
      }
      gamma <- as.vector(gamma_new)
      step_gamma <- step_gamma + 1
    }
    
    # beta
    step_beta <- 1
    while (step_beta <= irls_max_iter) {
      mu_curr <- exp(X %*% beta)
      
      # W_beta
      W_beta <- as.vector(z_hat_2 * mu_curr)
      W_beta <- pmax(W_beta, 1e-10) 
      
      # v_beta)
      v_beta <- X %*% beta + (z_hat_2 * (Y - mu_curr)) / W_beta
      
      # beta
      beta_new <- solve(t(X) %*% (W_beta * X)) %*% t(X) %*% (W_beta * v_beta)
      
      if (max(abs(beta_new - beta)) < irls_tol) {
        beta <- as.vector(beta_new)
        break
      }
      beta <- as.vector(beta_new)
      step_beta <- step_beta + 1
    }
    
    iter <- iter + 1
  }
  
  
  list(
    beta = as.vector(beta),
    gamma = as.vector(gamma),
    Q_value = ll_new,
    Q_history = loglik_history,
    em_iterations = min(iter, max_iter),
    z_hat_1 = z_hat_1,
    z_hat_2 = z_hat_2
  )
}
  
