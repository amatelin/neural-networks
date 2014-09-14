##To install the package : 
# source("http://bioconductor.org/biocLite.R")
# biocLite("EBImage")
library(EBImage)
library(neuralnet)


#######################HELPER FUNCTIONS####################################
image.vectorize <- function(x){
  dim <- ncol(x)*nrow(x)
  out <- matrix(ncol = dim)
  
  for (i in 1:nrow(x)){
    from = (i-1)*ncol(x)+1
    to = from + ncol(x)-1
    out[1, from:to] <- x[i,]
  }
  return(out)
}

image.process <- function(filepath) {
  image <- readImage(filepath)
  single.frame <- getFrame(image, 1)
  values.matrix <- imageData(single_frame)
  values.vector <- image.vectorize(values_matrix)
  
  return(values.vector)
}

nn.accuracy <- function(nn, in.test, out.ideal, rep) {
  ncols = ncol(out.ideal)
  nrows = nrow(out.ideal)
  out.actual <- matrix(ncol = ncols, nrow=nrows)
  correct.answers <- 0
  
  for (i in 1:nrow(in.test)){
    out.actual[i,] <- round(compute(nn, in.test[i,], rep=rep)$net.result, 3)
    for (j in 1:ncols){
      out.actual[i,j]<- ifelse(j==which.max(out.actual[i,]), 1, 0)
    }
    if (which.max(out.actual[i,]) == which.max(out.ideal[i,])){
      correct.answers <- correct.answers + 1
    }
  }
  
  correct.percentage <- round((correct.answers/nrows)*100, 2)
  return(correct.percentage)
}


nn.benchmark<- function(formula, training.data, testing.data, out.ideal, min.nodes.1layer, max.nodes.1layer, min.nodes.2layer, max.nodes.2layer, threshold, reps) {
  nrows <- ((max.nodes.1layer-min.nodes.1layer)+1)*((max.nodes.2layer-min.nodes.2layer)+1)
  
  benchmark.result <- matrix(ncol=6, nrow=nrows)
  result.row <- 1
  
  nn.results <- matrix(nrow=1, ncol=reps)
  
  for (i in min.nodes.1layer:max.nodes.1layer) {
    for (j in min.nodes.2layer:max.nodes.2layer) {
      print(paste("running the network with", i, "node(s) on the 1st hidden layer and", j, "node(s) on the 2nd"))
      nn.tested <- neuralnet(formula, training.data, hidden=c(i,j), linear.output=FALSE, threshold=threshold, rep=reps)
      for (k in 1:reps){
        nn.results[1,k] <- nn.accuracy(nn.tested, testing.data, out.ideal, k)
      }
      
      benchmark.result[result.row,]<- c(i, j, mean(nn.results[1,]), max(nn.results[1,]), min(nn.results[1,]), sd(nn.results[1,]))
      result.row <- result.row+1
    }
  }
  
  benchmark.result <- as.data.frame(benchmark.result)
  colnames(benchmark.result) <- c("layer 1", "layer 2", "average", "max", "min", "standard deviation")
  
  return(benchmark.result)
  
}





########ATTEMPT TO CLASSIFY CiRCLES (plain vs empty)######################
inputs <- matrix(nrow=20, ncol=1024)

for (i in 1:10) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/empty dot vs plain dot/training/empty_dot", as.character(i), ".png", sep="")
  inputs[i,] <- image.process(file)
}

for (i in 1:10) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/empty dot vs plain dot/training/full_dot", as.character(i), ".png", sep="")
  inputs[(i+10),] <- image.process(file)
}

output <- matrix(c(rep(0, 10), rep(1, 10)))

df <- data.frame(input = inputs, output=output)

library(neuralnet)

formula <- "output~input.1"
for (i in 2:ncol(inputs)){
  input <- paste("input.", i, sep="")
  formula <- paste(formula, input, sep="+")
}


nn <- neuralnet(formula, df, hidden=11, linear.output=FALSE, threshold=0.01)


######TESTING######
inputs_test <- matrix(nrow=10, ncol=1024)

for (i in 1:5) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/empty dot vs plain dot/testing/test_empty_dot", as.character(i), ".png", sep="")
  inputs_test[i,] <- image.process(file)
}

for (i in 1:5) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/empty dot vs plain dot/testing/test_full_dot", as.character(i), ".png", sep="")
  inputs_test[(i+5),] <- image.process(file)
}

df_test <- data.frame(input = inputs_test)

for (i in 1:10){
  prediction <- compute(nn, df_test[i,], rep=1)
  print(round(prediction$net.result, 0))
}








#################ATTEMPT TO CLASSIFY SHAPES##################
in.train <- matrix(nrow=45, ncol=1024)

##Loading rectangles
for (i in 1:15) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/training/rectangle", as.character(i), ".png", sep="")
  in.train[i,] <- image.process(file)
}

##Loading trangles
for (i in 1:15) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/training/triangle", as.character(i), ".png", sep="")
  in.train[i+15,] <- image.process(file)
}

##Loading circles
for (i in 1:15) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/training/circle", as.character(i), ".png", sep="")
  in.train[i+30,] <- image.process(file)
}

##1 for rectangle, 2 for triangle and 3 for circle
out1 <- matrix(c(rep(1,15), rep(0,30)))
out2 <- matrix(c(rep(0,15), rep(1,15), rep(0,15)))
out3 <- matrix(c(rep(0,30), rep(1,15)))


df.train <- data.frame(input=in.train, out1=out1, out2=out2, out3=out3)


formula <- "out1+out2+out3~input.1"
for (i in 2:ncol(inputs)){
  input <- paste("input.", i, sep="")
  formula <- paste(formula, input, sep="+")
}

nn <- neuralnet(formula, df.train, hidden=c(10,10), linear.output=FALSE, threshold=0.0001, rep=20)



###################################NN TEST############################################################
in.test <- matrix(nrow=24,  ncol=1024)
##Loading rectangles
for (i in 1:8) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/testing/rectangle", as.character(i), ".png", sep="")
  in.test[i,] <- image.process(file)
}

##Loading trangles
for (i in 1:8) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/testing/triangle", as.character(i), ".png", sep="")
  in.test[i+8,] <- image.process(file)
}

##Loading circles
for (i in 1:8) {
  file = paste("C:/Users/alexis.matelin/Documents/Neural Networks/Visual classification/shapes/testing/circle", as.character(i), ".png", sep="")
  in.test[i+16,] <- image.process(file)
}

df.test <- data.frame(input=in.test)

out1 <- c(rep(1,8), rep(0,16))
out2 <- c(rep(0,8), rep(1,8), rep(0,8))
out3 <- c(rep(0,16), rep(1,8))

out.ideal <- matrix(c(out1, out2, out3), ncol=3)


for (i in 1:24){
  prediction <- compute(nn, df.test[i,], rep=1)
  print(round(prediction$net.result, 2))
}

nn
