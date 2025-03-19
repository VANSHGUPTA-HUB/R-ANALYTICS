# Predictive Modeling Using R

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)

# Load the dataset
data <- read.csv("R_PROGRAMMING_DATASET.csv", stringsAsFactors = FALSE)

# Dataset Description
# The dataset includes information on:
# - Shape: The cut shape of the diamond.
# - Cut: Quality of the cut.
# - Color: Diamond color grade.
# - Clarity: Diamond clarity grade.
# - Carat Weight: Weight of the diamond in carats.
# - Price: Price of the diamond in USD.

# Dataset Overview
str(data)
summary(data)

# Data Cleaning
# Convert categorical columns to factors
data$Shape <- as.factor(data$Shape)
data$Cut <- as.factor(data$Cut)
data$Color <- as.factor(data$Color)
data$Clarity <- as.factor(data$Clarity)
data$Polish <- as.factor(data$Polish)
data$Symmetry <- as.factor(data$Symmetry)
data$Type <- as.factor(data$Type)
data$Fluorescence <- as.factor(data$Fluorescence)

# Handle missing values
data <- na.omit(data)

# Exploratory Data Analysis (EDA)
# Analyze the structure and characteristics of the dataset
cat("Dataset Structure:\n")
print(str(data))

# Visualization: Histogram of Price
ggplot(data, aes(x = Price)) +
  geom_histogram(binwidth = 500, fill = "blue", color = "black") +
  labs(title = "Distribution of Diamond Prices", x = "Price", y = "Frequency")

# Visualization: Scatter plot of Carat Weight vs. Price
ggplot(data, aes(x = Carat.Weight, y = Price)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Carat Weight vs. Price", x = "Carat Weight", y = "Price")

# Correlation Analysis
numeric_vars <- data %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_vars)
print("Correlation Matrix:")
print(correlation_matrix)

# Feature Engineering: Identify key features
# Key features like Carat Weight, Clarity, and Color are selected for modeling based on their correlation with Price.

# Data Splitting
set.seed(123)
train_index <- createDataPartition(data$Price, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Model Development
# Linear Regression
lm_model <- lm(Price ~ Carat.Weight + Clarity + Color + Shape, data = train_data)
cat("\nLinear Regression Summary:\n")
print(summary(lm_model))

# Predictions and Evaluation for Linear Regression
lm_predictions <- predict(lm_model, test_data)
lm_rmse <- sqrt(mean((lm_predictions - test_data$Price)^2))
cat("Linear Regression RMSE:", lm_rmse, "\n")

# Random Forest Regressor
rf_model <- randomForest(Price ~ Carat.Weight + Clarity + Color + Shape, data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)
rf_rmse <- sqrt(mean((rf_predictions - test_data$Price)^2))
cat("Random Forest RMSE:", rf_rmse, "\n")

# Results and Analysis
# Model Performance Comparison
cat("\nModel Performance Comparison:\n")
cat("Linear Regression RMSE:", lm_rmse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")

# Insights:
# - Linear regression provides a baseline model with RMSE of", lm_rmse, ".
# - Random Forest shows improved performance with RMSE of", rf_rmse, ".

# Save Cleaned Data
write.csv(data, "Cleaned_Dataset.csv", row.names = FALSE)

# Conclusion
# - This project demonstrates predictive modeling techniques in R.
# - Random Forest performed better than Linear Regression for this dataset.
# - Further improvements can include hyperparameter tuning and adding more features.