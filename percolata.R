  train_data <- read.csv('C:/percolata/train_data.csv',header = T,sep=',')
  str(train_data)
  train_data$device_angle <- as.factor(train_data$device_angle)
  train_data$distance_to_door <- as.factor(train_data$distance_to_door)
  train_data$AM_or_PM <- as.factor(train_data$AM_or_PM)
  train_data$mall_or_street <- as.factor(train_data$mall_or_street)
  
  sum(train_data$video_walkin==0)
  sum(train_data$predict_walkin==0)
  sum(train_data$wifi_walkin==0)
  
  
  new_data <- train_data[train_data$video_walkin!=0,]
  
  
  ###clustering
  
  library(dummies)
  clust.data.new <- dummy.data.frame(new_data)
  y <- as.matrix(clust.data.new)
  hc <- hclust(d=dist(y),method = 'average')
  
  
  dev.new()
  par(mfrow=c(1,1))
  plot(hc,cex=.1)
  dev.off()
  
  
  
  hc_cluster<- cutree(hc,k=2)
  new_data$cluster1 <- hc_cluster
  
  
  par(mfrow=c(1,1))
  with(new_data,boxplot(new_data$groundtruth_walkin~new_data$cluster1,xlab='cluster'))
  
  with(new_data,boxplot(new_data$video_walkin~new_data$cluster1,xlab='cluster'))
  with(new_data,boxplot(new_data$predict_walkin~new_data$cluster1,xlab='cluster'))
  with(new_data,boxplot(new_data$wifi_walkin~new_data$cluster1,xlab='cluster'))
  with(new_data,boxplot(new_data$sales_in_next_15_min~new_data$cluster1,xlab='cluster'))
  with(new_data,boxplot(new_data$sales_in_next_15_to_30_min~new_data$cluster1,xlab='cluster'))
  with(new_data,boxplot(new_data$average_person_size~new_data$cluster1,xlab='cluster'))
  
  clust1 <- new_data[new_data$cluster1==1,]
  clust2 <- new_data[new_data$cluster1==2,]
  
  sum(abs(clust1$groundtruth_walkin-clust1$video_walkin))
  sum(abs(clust2$groundtruth_walkin-clust2$video_walkin))
  
  ## the higher the avg.person size the closer video walk-in is to ground truth.
  
  table(new_data$cluster1,new_data$distance_to_door)
  
  train_data$diff <- abs(train_data$groundtruth_walkin-train_data$video_walkin)
  aggregate(train_data$diff~train_data$distance_to_door,train_data,mean)
  aggregate(train_data$diff~train_data$device_angle,train_data,mean)
  aggregate(train_data$diff~train_data$AM_or_PM,train_data,mean)
  aggregate(train_data$diff~train_data$mall_or_street,train_data,mean)
  with(train_data,boxplot(train_data$diff~train_data$average_person_size))
  
  
  aggregate(train_data$average_person_size~train_data$device_angle,train_data,mean)
  aggregate(train_data$average_person_size~train_data$distance_to_door,train_data,mean)
  aggregate(train_data$average_person_size~train_data$mall_or_street,train_data,mean)
  aggregate(train_data$average_person_size~train_data$AM_or_PM,train_data,mean)
  
  
  table(train_data$device_angle,train_data$mall_or_street)
  table(train_data$distance_to_door,train_data$mall_or_street)
  
  ## most of the cameras in the mall are kept at 2- distance to door.
  # this yielded a bigger average person size at mall than in streets which yielded better accuracy
  
  cor(new_data$groundtruth_walkin,new_data$sales_in_next_15_min)
  cor(new_data$groundtruth_walkin,new_data$sales_in_next_15_to_30_min)
  
  cor(new_data$video_walkin,new_data$groundtruth_walkin)
  cor(new_data$wifi_walkin,new_data$groundtruth_walkin)
  cor(new_data$predict_walkin,new_data$groundtruth_walkin)
  
  cor(new_data$video_walkin,new_data$video_walkout)
  cor(new_data$predict_walkin,new_data$predict_walkout)
  cor(new_data$wifi_walkin,new_data$wifi_walkout)
  
  ## video form has been the best to predict ground truth
  # more walk in accounted for more transactions
  
  hist(new_data$average_person_size)
  hist(new_data$video_walkin)
  
  
  #1) linear regression
  
  ## subset selection
  
  
  
  set.seed(1)
  train <- sample(1:nrow(train_data), .75*nrow(train_data))
  test <- (-train)
  train_data1 <- train_data[train,]
  test_data1 <- train_data[test,]
  
  attach(train_data1)
  rem <- c("cluster1","wifi_walkout","predict_walkout","video_walkout","AM_or_PM","device_angle",'diff')
  subset.data <- train_data1[ , !(names(train_data1) %in% rem)]
  m1 <- lm(subset.data$groundtruth_walkin~.,data = subset.data)
  summary(m1)
  
  pred.m1 <- predict(m1,test_data1)
  pred.m1[pred.m1<=0] <- 0
  pred.m1 <- ceiling(pred.m1)
  err.m1 <- sum(abs(test_data1$groundtruth_walkin-pred.m1))
  base.err <- sum(abs(test_data1$groundtruth_walkin - test_data1$video_walkin))
  ## the linear regression model was better than the base model.
  
  library(caret)
  
  ##random forest
  
  train_data1 <- train_data1[,-15]
  attach(train_data1)
  
  system.time(Mod1 <- train( groundtruth_walkin ~ ., method = "rf", 
                                           data = train_data1, importance = T, 
                                           trControl = trainControl(method = "cv", number = 3)))
  
  
  vi <- varImp(Mod1)
  vi$importance[1:10,]
  
  
  pred.Mod1 <- predict(Mod1,test_data1)
  pred.Mod1[pred.Mod1<=0] <- 0
  pred.Mod1 <- ceiling(pred.Mod1)
  as.vector(pred.Mod1) 
  
  err.Mod1 <- sum(abs(test_data1$groundtruth_walkin-pred.Mod1))
  base.err <- sum(abs(test_data1$groundtruth_walkin - test_data1$video_walkin))
  
  ## Random forest is clearly better
  
  
  library(randomForest)
  set.seed(1001)
  
  
  trees <- seq(5,200,by=10)
  error <- rep(0,length(trees))
  
  
  for(i in 1:length(trees)){
    mod.bag =randomForest(groundtruth_walkin ~ .,data=train_data1,mtry=13,ntree=trees[i], importance=TRUE)
    pred.bag=predict(mod.bag,newdata=test_data1)
    error[i] <- sum(abs(test_data1$groundtruth_walkin - pred.bag))
  }
  
  trees[which.min(error)]
  
  plot(trees,error,type='b',main='Bagging error with different tree sizes')
  
  set.seed(1001)
  mod.bag1=randomForest(groundtruth_walkin ~ .,data=train_data1,mtry=13,ntree=135, importance=TRUE)
  pred.bag1=predict(mod.bag,newdata=test_data1)
  pred.bag1[pred.bag1<=0] <- 0
  pred.bag1 <- ceiling(pred.bag1)
  as.vector(pred.bag1) 
  
  error.bag1 <- sum(abs(test_data1$groundtruth_walkin - pred.bag1))
  
  ## random forests
  trees <- seq(5,200,by=10)
  error <- matrix(rep(0,length(trees)*13),nrow=13)
  
  for(j in 1:13){
  for(i in 1:length(trees)){
    mod.bag =randomForest(groundtruth_walkin ~ .,data=train_data1,mtry=j,ntree=trees[i], importance=TRUE)
    pred.bag=predict(mod.bag,newdata=test_data1)
    error[j,i] <- sum(abs(test_data1$groundtruth_walkin - pred.bag))
  }
  }
  
  
  ## boosted trees
  library(gbm)
  set.seed(1001)
  nt <- seq(1,13,1)
  lr <- c(.1,.05,.01,.005,.001)
  trees <- c(1000,1500,2000,2500,3000,4000,5000)
  
  error.boost <- matrix(rep(0,1),nrow = 1)
  #error.boost <- rep(0,4)
  start.time <- Sys.time()
  
  for(i in lr){
    for(j in nt){
      for(k in trees){
  boost.mod <- gbm(groundtruth_walkin ~.,data = train_data1,distribution = 'gaussian',n.trees= k,interaction.depth = 4,shrinkage = i,cv.folds = 10)
  pred.boost <- predict.gbm(boost.mod,test_data1,n.trees = 1000 )
  error.boost1 <- sum(abs(test_data1$groundtruth_walkin - pred.boost))
  print(c(error.boost1,i,j,k))
  error.boost = rbind(error.boost1,error.boost)
      }
    }
  }
  end.time <- Sys.time()
  time <- end.time - start.time
  
  test_data <- read.csv('C:/percolata/test_data.csv',header = T,sep=',')
  
  set.seed(1001)
  start.time <- Sys.time()
  boost.mod <- gbm(groundtruth_walkin ~.,data = train_data1,distribution = 'gaussian',n.trees= 5000,interaction.depth = 9,shrinkage = .05,cv.folds = 10)
  pred.boost <- predict.gbm(boost.mod,test_data,n.trees = 5000 )
  error.boost <- sum(abs(test_data1$groundtruth_walkin - pred.boost))
  end.time <- Sys.time()
  time <- end.time - start.time
  test_data$groudtruth_walkin <- ceiling(pred.boost)
  test_data$groudtruth_walkin[test_data$groudtruth_walkin < 0] <- 0
  
  write.csv(test_data, 'C:/percolata/test_data1.csv')
  
  ## neural nets
  library(nnet)
  library(neuralnet)
  library(caret)
  library(Rcolorbrewer)
  
  
  scaled <- as.data.frame(scale(train_data[], center = mins, scale = maxs - mins))
  
  
  maxs <- apply(train_data[,5:13], 2, max) 
  mins <- apply(train_data[,5:13], 2, min)
  
  scaled <- as.data.frame(scale(train_data[,5:13], center = mins, scale = maxs - mins))
  train_data_sc <- cbind(train_data[,1:4],scaled)
  
  set.seed(1)
  train <- sample(1:nrow(train_data_sc), .75*nrow(train_data_sc))
  test <- (-train)
  train_data1_sc <- train_data_sc[train,]
  test_data1_sc <- train_data_sc[test,]
  
  train_data2 <- model.matrix(~groundtruth_walkin+device_angle+distance_to_door+AM_or_PM+
                                mall_or_street+average_person_size+video_walkin+predict_walkin+
                                wifi_walkin+sales_in_next_15_min+sales_in_next_15_to_30_min,train_data1_sc)
  
  start.time <- Sys.time()
  nn.mod <- neuralnet(groundtruth_walkin ~ device_angle2+device_angle3+distance_to_door2+
                        AM_or_PM1+mall_or_street2+average_person_size+video_walkin+predict_walkin+
                        wifi_walkin+sales_in_next_15_min+sales_in_next_15_to_30_min,
                                            data = train_data2,
                                            hidden = 8,
                                            rep=2,
                                            algorithm = 'backprop',
                                            err.fct = 'sse',
                                            linear.output = T,
                                            learningrate = .03)
  
  end.time <- Sys.time()
  run.time <- end.time - start.time
  
  dev.new()
  plot(nn.mod)
  nn.mod$response
  dev.off()
  
  test_data2<- model.matrix(~ device_angle+distance_to_door+AM_or_PM+
                              mall_or_street+average_person_size+video_walkin+predict_walkin+
                              wifi_walkin+sales_in_next_15_min+sales_in_next_15_to_30_min,test_data1_sc)
  
  stay <- c('device_angle2','device_angle3','distance_to_door2',
              'AM_or_PM1','mall_or_street2','average_person_size','video_walkin','predict_walkin',
              'wifi_walkin','sales_in_next_15_min','sales_in_next_15_to_30_min')
  test_data3<- test_data2[ , (colnames(test_data2) %in% stay)]
  
  nn.pred <- compute(nn.mod,as.data.frame(test_data3))
  names(nn.pred)
  class(nn.pred$net.result)
  
  train_data
