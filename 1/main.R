# Importar Dataset

data = read.csv('./Data.csv')

data$Age = ifelse(is.na(data$Age),ave(data$Age, FUN =  function(x) mean(x,na.rm = TRUE)),data$Age)
data$Salary = ifelse(is.na(data$Salary),ave(data$Salary, FUN =  function(x) mean(x,na.rm = TRUE)),data$Salary)


data$Country = factor(data$Country,levels = c("France","Spain","Germany"),labels = c(1,2,3))
data$Purchased = factor(data$Purchased,levels = c("No","Yes"),labels = c(0,1))

# Dividir training y testing del dataset
library(caTools)

set.seed(123)
split = sample.split(data$Purchased,SplitRatio = 0.8) 
training_set = subset(data,split == TRUE)
testing_set = subset(data,split == FALSE)

# Escalado de valores
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])

