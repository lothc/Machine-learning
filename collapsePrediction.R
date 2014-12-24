# by Christophe Loth
# Written on June 11, 2013
#
# This code fits multivariate supervised learning models (logistic regression, Naive Bayes, 
# SVM with various kernels, Neural network, ...) to predict structural collapse based on
# measures of ground motion intensity. A 2D example is shown and illustrated at the end. For simplicity, 
# no train/test split of the data is performed.
#
#
# Inputs:
#   - a n-by-p data frame associating n vector-valued spectral accelerations (p periods) to 
#     collapse/non-collapse structural analyses (1/0). The first p columns are the predictor variables,
#     the (p+1)th column is the collapse response
#
# Output:
#   - List of fitted models
#      $logRegModel
#      $naiveBayes
#      $svmModelLinear
#      $svmModelGaussian
#      $nnet2


rm(list = ls())



library(e1071)
library(nnet)
library(rpart)
#library(kernlab)


set.seed(100)


fitModels=function(collapseData){
  # Create matrix of predictors and response vector
  p=dim(collapseData)[2]-1
  x=as.matrix(collapseData[,1:p])
  y=as.matrix(collapseData[,p+1])
  
  # Fit various models and output in list object
  modelList=list()
  
  modelList$logRegModel = glm( y ~ x + 1 , family=binomial(link = "logit"))
  modelList$svmModelLinear = svm( x , y , type='C-classification', 
                                 kernel='linear',scale=FALSE)
  modelList$svmModelGaussian = svm( x , y , type='C-classification', 
                                   kernel='radial',scale=FALSE)
  modelList$nnet2 = nnet( x , y , size=3)
  modelList$nB=naiveBayes( x , as.factor(y)) 
  


  return(modelList)
  
  
}






# Example use

# Generate data (p=2)
n=1000
collapseData=as.data.frame(matrix(ncol=3,nrow=n))

for (i in 1:2){
  collapseData[,i]=runif(n,0,2)
}
collapseData[,3]=1/(1+exp(-2*log(collapseData[,1])-4*log(collapseData[,2]))) +rnorm(n,0,0.3)
collapseData[,3]=sapply(collapseData[,3],function(x){if (x>=0.5){1} else {0}})

x=as.matrix(collapseData[,1:2])
y=as.matrix(collapseData[,3])
maskCol=(y==1)

# Fit the models
modelList=fitModels(collapseData)
attach(modelList)



# Plot the results
source('filled.contour3.R')
# Custom function to plot nonlinear decision boundaries
plotFun=function(model){
  zpred=predict(model,as.matrix(xMatPlot))
  zpred=matrix(as.numeric(as.vector(zpred)),nrow=length(xplot),ncol=length(xplot))
  plot.new()
  filled.contour3(xplot,xplot,zpred,nlevels=2,col=rev(cm.colors(n=2, alpha=0.5)),  
                  plot.axes={axis(1,cex.axis=1.5) 
                             axis(2,cex.axis=1.5)},
                  plot.title={title(xlab='Ground motion intensity 1',cex.lab=1.5,
                                    ylab='Ground motion intensity 2')}
  )
  points(x[maskCol,1],x[maskCol,2],pch=4,col="red",cex=0.75)
  points(x[!maskCol,1],x[!maskCol,2],pch=1,col="green",cex=0.75)
  
}

pdf('fittedModels.pdf',width=18,height=12)
par(mfrow=c(2,3),mar=c(5,5,5,5))


plot(x[,1],x[,2],type="n",xlim=c(0,2),ylim=c(0,2),xlab='Ground motion intensity 1',
     ylab='Ground motion intensity 2', xaxs = "i", yaxs = "i",cex.axis=1.5,cex.lab=1.5,cex=1.5)
maskCol=(y==1)
points(x[maskCol,1],x[maskCol,2],pch=4,col="red",cex=0.75)
points(x[!maskCol,1],x[!maskCol,2],pch=1,col="green",cex=0.75)
title(main='Collapse data',cex.main=2)





xplot=seq(0,2,0.01)
xMatPlot=expand.grid(xplot,xplot)
names(xMatPlot)=c('V1','V2')
# SVM with linear kernel + logistic regression 
plotFun(svmModelLinear)
title(main='Logistic regression + SVM with linear kernel',cex.main=2)

lines(xplot,-1/logRegModel$coeff[[3]]*(logRegModel$coeff[1]+logRegModel$coeff[2]*xplot),
      lty=2,lwd=3,col="skyblue3")

# support vectors
x1s=x[,1]
x2s=x[,2]
x1min = min(x1s); x1max = max(x1s);
x2min = min(x2s); x2max = max(x2s);
coef1 = sum(svmModelLinear$coefs*x1s[svmModelLinear$index]);
coef2 = sum(svmModelLinear$coefs*x2s[svmModelLinear$index]);
lines(c(x1min,x1max),  (svmModelLinear$rho-coef1*c(x1min, x1max))/coef2)
lines(c(x1min,x1max),  (svmModelLinear$rho+1-coef1*c(x1min, x1max))/coef2, lty=2)
lines(c(x1min,x1max),  (svmModelLinear$rho-1-coef1*c(x1min, x1max))/coef2, lty=2)



# Naive Bayes
plotFun(nB)
title(main='Naive Bayes',cex.main=2)


# SVM with Gaussian kernel
plotFun(svmModelGaussian)
title(main='SVM with Gaussian kernel',cex.main=2)

# Neural network of size 2
plotFun(nnet2)
title(main='Neural network with 1 hidden layer of 3 units',cex.main=2)




dev.off()
