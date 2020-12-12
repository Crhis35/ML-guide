
# Regresion Lineal simple

dataset = read.csv("./Salary_Data.csv")

library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Escalado de valores
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

# Ajustar el modelo de regresion lineal simple con el conjunto de entrenamiento

regresor = lm(formula = Salary ~ YearsExperience,data = training_set)

# Predecir el conjunto de resultado de tes
y_pred = predict(regresor, newdata = testing_set)


library(ggplot2)
# Visualizar los datos conjuntos de entrenamiento

ggplot() +
    geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),colour= "red") + 
    geom_line(aes(x = training_set$YearsExperience, y = predict(regresor, newdata = training_set)))+
    ggtitle("Sueldo VS Años de experiencia [Conjunto de entrenamiento]") +
    xlab("Años de experiencia") + 
    ylab("Sueldo (en $) ")
# Visualizar los datos conjuntos de testing

ggplot() +
    geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),colour= "red") + 
    geom_line(aes(x = testing_set$YearsExperience, y = predict(regresor, newdata = testing_set)))+
    ggtitle("Sueldo VS Años de experiencia [Conjunto de Test]") +
    xlab("Años de experiencia") + 
    ylab("Sueldo (en $) ")
    

