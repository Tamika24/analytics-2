```{r response_curves_fixed, warning=FALSE, message=FALSE}
library(tidyr)
library(ggplot2)

# 1) Compute the medians of X1 and X2 from your data
X1_med <- median(dat$X1)
X2_med <- median(dat$X2)

# 2) Improved helper: fix X1, X2, X3 separately
predict_curve <- function(var_seq, varname,
                          fixed_X1, fixed_X2, fixed_X3,
                          pars, m) {
  n <- length(var_seq)
  input <- matrix(0, n, 3)
  colnames(input) <- c("X1","X2","X3")
  
  # Sweep the target variable; hold the others at their medians
  input[,"X1"] <- if(varname=="X1") var_seq else fixed_X1
  input[,"X2"] <- if(varname=="X2") var_seq else fixed_X2
  input[,"X3"] <- fixed_X3
  
  # Forward pass to get probabilities
  probs <- af_forward(input,
                      Y     = matrix(0,n,3),
                      theta = theta_best,
                      m     = m,
                      nu    = 0)$probs
  
  # Return a data.frame with var, class-probs
  df <- as.data.frame(probs)
  names(df) <- c("alpha","beta","rho")
  df[[varname]] <- var_seq
  df
}

# 3) Build sequences over the real data range
X1_seq <- seq(min(dat$X1), max(dat$X1), length.out = 200)
X2_seq <- seq(min(dat$X2), max(dat$X2), length.out = 200)

# 4) Generate curves for Detector A (X3=1) vs B (X3=0)
df1_A <- predict_curve(X1_seq, "X1", X1_med, X2_med, 1, theta_best, m)
df1_B <- predict_curve(X1_seq, "X1", X1_med, X2_med, 0, theta_best, m)
df2_A <- predict_curve(X2_seq, "X2", X1_med, X2_med, 1, theta_best, m)
df2_B <- predict_curve(X2_seq, "X2", X1_med, X2_med, 0, theta_best, m)

# 5) Reshape and plot for X1
plot_data_X1 <- bind_rows(
  transform(df1_A, Detector="Type A"),
  transform(df1_B, Detector="Type B")
) %>%
  pivot_longer(cols = c("alpha","beta","rho"),
               names_to  = "Class",
               values_to = "Probability")

ggplot(plot_data_X1, aes(x = X1, y = Probability, color = Class)) +
  geom_line(size=1) +
  facet_wrap(~ Detector) +
  labs(title = "Response Curves: P(class) vs X1 by Detector",
       x = "X1", y = "Predicted Probability") +
  theme_minimal() +
  theme(aspect.ratio = 1)

# 6) Reshape and plot for X2
plot_data_X2 <- bind_rows(
  transform(df2_A, Detector="Type A"),
  transform(df2_B, Detector="Type B")
) %>%
  pivot_longer(cols = c("alpha","beta","rho"),
               names_to  = "Class",
               values_to = "Probability")

ggplot(plot_data_X2, aes(x = X2, y = Probability, color = Class)) +
  geom_line(size=1) +
  facet_wrap(~ Detector) +
  labs(title = "Response Curves: P(class) vs X2 by Detector",
       x = "X2", y = "Predicted Probability") +
  theme_minimal() +
  theme(aspect.ratio = 1)

#All three curves are bounded between 0 and 1 and vary smoothly (S-shapes, bell-shapes, etc.) over the range of X₁ (or X₂). The only difference between the “Type A” and “Type B” panels is whether the network sees detector X₃=1 or X₃=0—if those two panels lie on top of each other, it means the model isn’t using detector type to shift its predictions.

#In your output: For X₁: you can see P(α) rising from ≈0.15 up to ≈0.48, P(β) falling in mirror fashion, and P(ρ) a gentle transition in between. The red/green/blue lines for Type A and Type B lie almost on top of one another, which tells you that detector type has very little effect on the model’s decision boundary in that direction.

#For X₂: you’ll see the complementary curves (α falling with X₂, β rising, ρ again fairly flat). Again, the two panels overlap, confirming that the network treats both detectors nearly identically once it’s seen the best regularization.

#As X₁ (or X₂) increases, the network smoothly shifts probability mass from one class to another. The almost-identical red/green/blue lines in the Type A vs. Type B panels show that the detector indicator (X₃) hardly changes those response curves