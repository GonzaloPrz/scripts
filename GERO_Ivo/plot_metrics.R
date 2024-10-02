library(ggplot2)
library(dplyr)

# Define function for 95% confidence intervals
conf_int_95 <- function(data) {
  mean <- mean(data, na.rm = TRUE)
  inf <- quantile(data, probs = 0.025, na.rm = TRUE)
  sup <- quantile(data, probs = 0.975, na.rm = TRUE)
  return(c(mean, inf, sup))
}

# Tasks and y_label definition
tasks <- c('animales', 'fas')
y_label <- 'mean_absolute_error_MMSE_Total_Score'

# Define the results directory based on the OS
results_dir <- ifelse(grepl("gp", Sys.info()['user']),
                      file.path(Sys.getenv("HOME"), "results", "GERO_Ivo"),
                      "D:/results/GERO_Ivo")

# Load the metrics data (assuming it's a .pkl or .csv file)
metrics_df <- readRDS(file.path(results_dir, "all_metrics.rds"))

# Update the 'dimension' and 'task' columns with more descriptive labels
metrics_df <- metrics_df %>%
  mutate(dimension = recode(dimension,
                            "properties" = "DSMs (Word properties)",
                            "valid_responses" = "Manual assessment (valid responses)"),
         task = recode(task,
                       "animales" = "Semantic fluency",
                       "fas" = "Phonemic fluency"))

# Group by 'task' and 'dimension' and calculate mean, lower, and upper CI
summary_df <- metrics_df %>%
  group_by(task, dimension) %>%
  summarise(mean_score = mean(get(y_label), na.rm = TRUE),
            lower_ci = conf_int_95(get(y_label))[2],
            upper_ci = conf_int_95(get(y_label))[3]) %>%
  ungroup()

# Print the summary for verification
print(summary_df)

# Plot using ggplot2 in R
p <- ggplot(metrics_df, aes(x = task, y = get(y_label), fill = dimension)) +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.5, position = position_dodge(width = 0.5)) +
  geom_pointrange(aes(x = task, y = mean_score, ymin = lower_ci, ymax = upper_ci, color = dimension),
                  data = summary_df, 
                  position = position_dodge(width = 0.5), 
                  size = 1) +
  labs(x = "Task", y = "Mean absolute error in predicting MMSE scores") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_text(size = 12, family = "Calibri"),
        legend.title = element_text(size = 11, family = "Calibri"),
        legend.text = element_text(size = 9, family = "Calibri"))

# Display the plot
print(p)

# Save the plot
ggsave(filename = file.path(results_dir, paste0(y_label, "_comparisons.png")), plot = p, dpi = 300)
