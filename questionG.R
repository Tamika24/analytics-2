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
       col    = "red", lty = 2, pch = 16, bty = "n")set.seed(2025)
