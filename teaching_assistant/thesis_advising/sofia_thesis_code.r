# load in data
library(readr)
data <- read_csv("/Users/joshuaingram/Downloads/data.csv")

View(data)

cor(data$year, data$religiosity, na.rm = TRUE)

plot(data$year, data$religiosity)

fake <- data$religiosity

fake[1:3] <- fake[5]
fake[13:15] <- fake[12]
data$fake <- fake

plot(data$year, data$fake)

cor(data$year, data$fake)

# model fitting (linear regression)
model1 <- lm(pro_choice ~ women_in_congress + avg_social_media_users + fake, data = data)
summary(model1)

model_reduced <-  lm(pro_choice ~ women_in_congress + avg_social_media_users, data = data)
summary(model_reduced)

# partial F-test
anova(model_reduced, model1)