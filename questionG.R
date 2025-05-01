# 1. Split 80/20 (with seed)
set.seed(2025)
N           <- nrow(dat)
train_idx   <- sample(1:N, 0.8 * N, replace = FALSE)
valid_idx   <- setdiff(1:N, train_idx)

X_train     <- as.matrix(dat[train_idx, 1:3])
Y_train     <- as.matrix(dat[train_idx, 4:6])
X_valid     <- as.matrix(dat[valid_idx, 1:3])
Y_valid     <- as.matrix(dat[valid_idx, 4:6])

# 2. Network dimensions & parameter count
m      <- 4
p      <- ncol(X_train)
q      <- ncol(Y_train)
npars  <- 2*p^2 + 2*p + 2*p*m + 2*m + m^2 + m*q + q

# 3. Initialise random θ (as in lecture)
theta_rand <- runif(npars, -1, 1)

# 4. Grid of regularisation strengths
n_nu   <- 15
nu_seq <- exp(seq(-8, 2, length.out = n_nu))

# 5. Loop: fit on training, evaluate on validation
Val_error <- numeric(n_nu)
for (i in seq_len(n_nu)) {
  nu <- nu_seq[i]
  
  # penalised objective on training set
  obj_pen <- function(pars) {
    af_forward(X_train, Y_train, pars, m, nu)$obj
  }
  
  # fit using nlm()
  res_opt <- nlm(obj_pen, theta_rand, iterlim = 1000)
  
  # record unpenalised cross‐entropy on validation set
  Val_error[i] <- af_forward(
    X_valid, Y_valid,
    res_opt$estimate, m, 0
  )$loss
  
  cat("nu =", signif(nu,3),
      "→ val loss =", signif(Val_error[i],4), "\n")
}

# 6. Plot validation loss vs ν on log‐x
best_i  <- which.min(Val_error)
best_nu <- nu_seq[best_i]

plot(nu_seq, Val_error, type = "b", pch = 16,
     log  = "x",
     xlab = expression(nu),
     ylab = "Validation Cross‐Entropy Loss",
     main = "Validation Loss vs Regularization (m = 4)")
abline(v = best_nu, col = "red", lty = 2)
points(best_nu, Val_error[best_i], col = "red", pch = 16)
legend("topright",
       legend = paste0("Chosen ν = ", signif(best_nu,3)),
       col    = "red", lty = 2, pch = 16, bty = "n")


############################################################


# Step 2: Prepare the training and validation datasets
X_train <- as.matrix(train_data[, 1:3])  # Input features
Y_train <- as.matrix(train_data[, 4:6])  # Response variables (one-hot encoded)

X_valid <- as.matrix(valid_data[, 1:3])  # Input features
Y_valid <- as.matrix(valid_data[, 4:6])  # Response variables (one-hot encoded)

# Step 3: Define the objective function with regularization
v=0.01
bj_pen <- function(pars) {
  result <- af_forward(X_train, Y_train, theta, m, v)
  return(result$obj)
}

theta_rand = runif(npars,-1,1)
obj_pen(theta_rand)

res_opt = nlm(obj_pen,theta_rand,iterlim = 1000)
res_opt

# Step 4: Grid search over regularization parameter nu
n_nu <- 15
validation_errors = rep(NA,n_nu)
v_values <- exp(seq(-6, 2, length.out = n_nu))
#validation_errors <- numeric(length(v_values))

for (i in 1:n_nu) {
  v <- v_values[i]
  res_opt = nlm(obj_pen,theta_rand,iterlim = 1000)
  
  theta_rand <- runif(npars, -1, 1)
  
  # Fit the model using optim() to minimize the objective function
  fit <- optim(theta_rand, objective_fn, X = X_train, Y = Y_train, m = 4, v=v)
  
  # Get the predicted probabilities for validation set
  Yhat_valid <- af_forward(X_valid, Y_valid, fit$par, m = 4, v=v)$probs
  
  # Compute the validation error
  validation_errors[i] <- g(Yhat_valid, Y_valid)
}

# Step 5: Plot validation error vs nu
plot(v_values, validation_errors, type = "b", col = "blue", pch = 19, asp = 1, log="x", 
     xlab = "Regularization Parameter (ν)", ylab = "Validation Error",
     main = "Validation Error vs Regularization Parameter (ν)")
grid()

# Step 6: Choose the optimal regularization level (ν)
optimal_v <- v_values[which.min(validation_errors)]

