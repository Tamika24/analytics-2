set.seed(2025)
library(tidyr)
# Step 1: Split the data into training and validation sets (80%/20%)
n <- nrow(dat)
train_size <- floor(0.8 * n)
train_indices <- sample(1:n, train_size)
train_data <- dat[train_indices, ]
valid_data <- dat[-train_indices, ]

# Step 2: Prepare the training and validation datasets
X_train <- as.matrix(train_data[, 1:3])  # Input features
Y_train <- as.matrix(train_data[, 4:6])  # Response variables (one-hot encoded)

X_valid <- as.matrix(valid_data[, 1:3])  # Input features
Y_valid <- as.matrix(valid_data[, 4:6])

m        <- 4
p        <- ncol(X_train)
q        <- ncol(Y_train)
npars    <- 2*p^2 + 2*p + 2*p*m + 2*m + m^2 + m*q + q

# 2) Penalised objective (uses af_forward to return $obj)
nu      <- 0.01   # placeholder, will be reset in loop
obj_pen <- function(pars) {
  af_forward(X_train, Y_train, pars, m, nu)$obj
}

# 3) Prepare grid and storage
nu_seq        <- exp(seq(-6, 2, length = 20))
Val_error     <- numeric(length(nu_seq))    

theta_init <- runif(npars, -1, 1)
for(i in seq_along(nu_seq)) {
  nu <- nu_seq[i]
  
  res_opt <- nlm(
    f      = function(th) af_forward(X_train, Y_train, th, m, nu)$obj,
    p      = theta_init,
    iterlim= 1000
  )
  
  theta_init      <- res_opt$estimate           # ← warm-start next iteration
  Val_error[i]    <- af_forward(X_valid,
                                Y_valid,
                                res_opt$estimate,
                                m, 0)$loss
}

# 5) Plot on log‐x to reveal the U‐shape
best_i  <- which.min(Val_error)
best_nu <- nu_seq[best_i]

plot(nu_seq, Val_error, type="b", pch=16,
     log = "x",
     xlab = expression(nu),
     ylab = "Validation Cross‐Entropy Loss",
     main = "Validation Loss vs Regularization (m = 4)")
abline(v = best_nu, col = "red", lty = 2)
points(best_nu, Val_error[best_i], col = "red", pch = 16)
legend("topright",
       legend = paste0("chosen ν = ", signif(best_nu, 3)),
       col    = "red", lty = 2, pch = 16, bty = "n")

