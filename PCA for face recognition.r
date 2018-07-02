###############################################
####### PCA for human face reconstruction #####
###############################################


### LOAD LIBRARIES ###

library(pixmap)
library(readbitmap)
library(bmp)
library(stats)

### READ DATA ###

# Data consists of 200 human faces (already preprocessed and stored randomly)
# Each face is stored as a vector

data <-lapply(dir(), read.bmp)

#### SPLIT DATA INTO TRAINING AND TESTING SET  ###

# Allocate fist 150 faces as a training set
train <- data[1:150]
train <- lapply(train, unlist)
train <- lapply(train, matrix, nrow=256)
training <-array(0, dim=c(65536, 150))
for (i in 1:150){
training[,i] <- c(train[[i]])
}

# Create testing set with the remaining 50 faces
test<- data[151:200]
test<- lapply(test, unlist)
test<- lapply(test, matrix, nrow=256)
testing<-array(0, dim=c(65536, 50))
for (i in 1:50){
    testing[,i]<- c(test[[i]])
}

### COMPUTE THE MEAN FACE AND PLOT IT ###

avg.face <- apply(training, 1, mean)
avg.face.matrix <- matrix(avg.face, nrow=256, ncol=256)
avgFACE <- pixmapRGB(avg.face.matrix)
plot(avgFACE)

### OBTAIN EIGENFACES USING PCA AND PLOT THE FIRST 10 ###

# Substract the mean from the training set
S  <- apply(training, 2, function(x) x-avg.face)
# Covariance matrix C=SS' is too large, instead we get B=S'S which is much smaller
B <- t(S) %*% S
# u is a list containing eigenvalues and eigenvectors of B 
u <- eigen(B)
# v are the eigenvectors of C, but are not normalized
v <- S %*% u$vectors
# "norms" are the normalized eigenvectors
norms<- function(x) x/(sqrt( t(x) %*% x))
# Create the eigenfaces
eigenfaces <- apply(v, 2, norms)
eigenvalues<- u$values/sum(u$values)

# Plot the first 10 eigenfaces
eface1<-pixmapRGB(matrix(eigenfaces[,1], nrow=256, ncol=256))
eface2<-pixmapRGB(matrix(eigenfaces[,2], nrow=256, ncol=256))
eface3<-pixmapRGB(matrix(eigenfaces[,3], nrow=256, ncol=256))
eface4<-pixmapRGB(matrix(eigenfaces[,4], nrow=256, ncol=256))
eface5<-pixmapRGB(matrix(eigenfaces[,5], nrow=256, ncol=256))
eface6<-pixmapRGB(matrix(eigenfaces[,6], nrow=256, ncol=256))
eface7<-pixmapRGB(matrix(eigenfaces[,7], nrow=256, ncol=256))
eface8<-pixmapRGB(matrix(eigenfaces[,8], nrow=256, ncol=256))
eface9<-pixmapRGB(matrix(eigenfaces[,9], nrow=256, ncol=256))
eface10<-pixmapRGB(matrix(eigenfaces[,10], nrow=256, ncol=256))

par(mfrow=c(2,5), mar=c(0,0,0,0))
plot(eface1)
plot(eface2)
plot(eface3)
plot(eface4)
plot(eface5)
plot(eface6)
plot(eface7)
plot(eface8)
plot(eface9)
plot(eface10)

### RECONSTRUCT THE FACES IN THE TESTING SET AND PLOT THE FIRST FOUR ###

# Compute weights for test set using first 20 eigenvectors
short.eigen.face <- eigenfaces[,1:20]
ST <- apply(testing, 2, function(x) x - avg.face)
testing.w<- t(short.eigen.face) %*% ST

# Plot the first four reconstructed faces against the actual faces
newfaces <- avg.face + short.eigen.face%*%testing.w

# These are the four reconstructed faces
nface1<-pixmapRGB(matrix(newfaces[,1], nrow=256))
nface2<-pixmapRGB(matrix(newfaces[,2], nrow=256))
nface3<-pixmapRGB(matrix(newfaces[,3], nrow=256))
nface4<-pixmapRGB(matrix(newfaces[,4], nrow=256))
nface5<-pixmapRGB(matrix(newfaces[,5], nrow=256))

# These are the five actual faces in the test set
tface1<-pixmapRGB(test[[1]])
tface2<-pixmapRGB(test[[2]])
tface3<-pixmapRGB(test[[3]])
tface4<-pixmapRGB(test[[4]])
tface5<-pixmapRGB(test[[5]])

# Plot comparing reconstructed and actual faces
par(mfrow=c(2,4), mar=c(0,0,0,0))
plot(tface1)
plot(tface2)
plot(tface3)
plot(tface4)
plot(nface1)
plot(nface2)
plot(nface3)
plot(nface4)

### COMPUTE THE TOTAL RECONSTRUCTION ERROR ###

# Error is defined as the squared difference between reconstructed and original faces

# Define a distance function
distance <- function(a, b) { sum((a-b)^2)}

kk<-100
errors.average <- rep(0,kk)

for (k in 1:kk){
    short.eigen.face <- eigenfaces[,1:k]
    ST <- apply(testing, 2, function(x) x - avg.face)
    testing.w <- t(short.eigen.face) %*% ST
    newfaces <- avg.face+ short.eigen.face%*%testing.w

    errors <- rep(0,50)
    errors.sum <- rep(0, 50)
    for (i in 1:50){
        errors[i] <- distance(testing[,i], newfaces[,i])/65536
    }
    errors.average[k] <- mean(errors)
}

# Plot the errors
plot(errors.average, cex=0.5, pch=20)
