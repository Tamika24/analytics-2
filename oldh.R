m <- 4
v <- best_v
# Re-train model at best nu
obj_pen_best <- function(pars)
{
  af_forward(X_train, Y_train, pars, m, v)$obj
}

# Reinitialize random parameters
theta_rand <- runif(npars, -1, 1)

res_opt_best <- nlm(obj_pen_best, p = theta_rand, iterlim = 1000)
theta_best <- res_opt_best$estimate


# Plot response curves by varying X1 and X2 separately
# Helper function to predict probability curves
predict_curve <- function(var_seq, varname, fixed_X2 = 0, fixed_X3 = 0, pars, m) 
{
  n <- length(var_seq)
  input <- matrix(0, nrow = n, ncol = 3)
  colnames(input) <- c("X1", "X2", "X3")
  
  input[, "X1"] <- if (varname == "X1") var_seq else fixed_X2
  input[, "X2"] <- if (varname == "X2") var_seq else fixed_X2
  input[, "X3"] <- fixed_X3
  
  q <- 3 
  
  preds <- af_forward(input, Y = matrix(0, nrow=n, ncol=q), pars, m, v=0.                      )$probs
  
  out <- as.data.frame(preds)
  colnames(out) <- c("alpha", "beta", "rho")
  out[[varname]] <- var_seq
  
  return(out)
}

# Create sequences
X_seq <- seq(-4, 4, length.out = 100)

# Response curves for Detector Type A (X3=1) and Type B (X3=0)

curve_X1_A <- predict_curve(X_seq, "X1", fixed_X2=0, fixed_X3=1, theta_best, m)
curve_X1_B <- predict_curve(X_seq, "X1", fixed_X2=0, fixed_X3=0, theta_best, m)

curve_X2_A <- predict_curve(X_seq, "X2", fixed_X2=0, fixed_X3=1, theta_best, m)
curve_X2_B <- predict_curve(X_seq, "X2", fixed_X2=0, fixed_X3=0, theta_best, m)

# Helper to prepare data
prepare_plot_data <- function(curve_data, varname, type_label) 
{
  df <- as.data.frame(curve_data)
  colnames(df) <- c(varname, "alpha", "beta", "rho")
  df$Detector <- type_label
  df <- pivot_longer(df, cols = c("alpha", "beta", "rho"),
                     names_to = "Class", values_to = "Probability")
  return(df)
}

plot_data_X1 <- bind_rows(prepare_plot_data(curve_X1_A, "X1", "Type A"),
                          prepare_plot_data(curve_X1_B, "X1", "Type B"))

ggplot(plot_data_X1, aes(x = X1, y = Probability, color = Class)) +
  geom_line() +
  facet_wrap(~ Detector) +
  labs(title = "Predicted Class Probabilities vs X1",
       x = "X1", y = "Probability") +
  theme_minimal() +
  theme(aspect.ratio = 1)


plot_data_X2 <- bind_rows(prepare_plot_data(curve_X2_A, "X2", "Type A"),
                          prepare_plot_data(curve_X2_B, "X2", "Type B"))

ggplot(plot_data_X2, aes(x = X2, y = Probability, color = Class)) +
  geom_line() +
  facet_wrap(~ Detector) +
  labs(title = "Predicted Class Probabilities vs X2",
       x = "X2", y = "Probability") +
  theme_minimal() +
  theme(aspect.ratio = 1)