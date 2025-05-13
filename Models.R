calculate_metrics <- function(model_name, predictions, train_y, Recall, Precis, i) {
  table_result <- table(train_y, predictions, dnn = c("真实值", "预测值"))
  if(length(table_result) == 4){
    if (table_result[2, 1] != 0) {
      Recall[i, model_name] <<- table_result[2, 2] / table_result[2, 1]
    }
    if (table_result[1, 2] != 0) {
      Precis[i, model_name] <<- table_result[2, 2] / table_result[1, 2]
    }
  }
}
Methods = c('dire', 'buy', 'max', 'min')
nm=names(a0)

for (i in 1:4) {
  method = Methods[i]
  if (method=='buy') {
    bef=a0[(2*alth+1):(6*alth),-c(1:3)]
    tim=TIME[(2*alth+1):(6*alth)]
  }else{
    bef=a0[(2*alth+1):(5*alth),-c(1:3)]
    tim=TIME[(2*alth+1):(5*alth)]
  }
  names(bef)=nm[-c(1,2,3)]
  
  #Input
  x.temp <- switch(method,
                   "dire" = as.numeric(xn$现价 > 0.01 * tm),
                   "buy" = as.numeric(xn$最低 < (-0.01 * tm)),
                   "max" = as.numeric(xn$最高 > 0.02 * tm),
                   "min" = as.numeric(xn$最低 > 0.02 * tm)
  )
  if (method=='buy') {
    pre<-x.temp[(2*alth+1):(6*alth)]
    temp=cbind(xn[1:(4*alth),],xn1[1:(4*alth),],pre,bef,tim)
  }else{
    pre<-x.temp[(3*alth+1):(6*alth)]
    temp=cbind(xn[1:(3*alth),],xn1[1:(3*alth),],pre,bef,tim)
  }
  
  test=temp
  source("F:/Desktop/THS/Code/20240717/GetModel.R")
  
  #TEST_cm
  if (method=='buy') {
    bef=a0[(6*alth+1):(7*alth),-c(1:3)]
    names(bef)=nm[-c(1,2,3)]
    tim=TIME[(6*alth+1):(7*alth)]
    pre<-x.temp[(6*alth+1):(7*alth)]
    temp=cbind(xn[(4*alth+1):(5*alth),],xn1[(4*alth+1):(5*alth),],pre,bef,tim)
  }else{
    bef=a0[(5*alth+1):(6*alth),-c(1:3)]
    names(bef)=nm[-c(1,2,3)]
    tim=TIME[(5*alth+1):(6*alth)]
    pre<-x.temp[(6*alth+1):(7*alth)]
    temp=cbind(xn[(3*alth+1):(4*alth),],xn1[(3*alth+1):(4*alth),],pre,bef,tim)
  }
  
  
  test=temp
  source("F:/Desktop/THS/Code/20240717/TestModel.R")
  
  # 调用函数计算各模型的指标
  calculate_metrics("GLM", GLMpre, train_y, Recall, Precis, i)
  calculate_metrics("LM", LMpre, train_y, Recall, Precis, i)
  calculate_metrics("NB", NBpre, train_y, Recall, Precis, i)
  calculate_metrics("XG", XGpre, train_y, Recall, Precis, i)
  calculate_metrics("RF", RFpre, train_y, Recall, Precis, i)
  
  #Predict
  bef=a0[(7*alth+1):(8*alth),-c(1:3)]
  names(bef)=nm[-c(1,2,3)]
  tim=TIME[(7*alth+1):(8*alth)]
  temp=cbind(xn[(5*alth+1):(6*alth),],xn1[(5*alth+1):(6*alth),],rep(1,1*alth),bef,tim)
  
  test=temp
  source("F:/Desktop/THS/Code/20240717/UseModel.R")
  
  switch(method,
         "dire" = {
           DIRE=data.frame(ID=a0[1:alth,c(1:4)],LM=lmpre,GLM=glmpre,NB=NBpre,XG=XGpre,RF=RFpre)
           rate=(a0[(lth-alth+1):lth,"现价"]-tavg[(lth-alth+1):lth])/a0[(lth-alth+1):lth,"现价"]
           PDATA=data.frame(ID=a0[(lth-alth+1):lth,c(1:4,17)],IP=ipan,OP=opan,uIP=l.ipan,uOP=l.opan,Stavg=stavg,Safety=rate,LM=lmpre,GLM=glmpre,NB=NBpre,XG=XGpre,RF=RFpre)
         },
         "buy" = {
           BUY=data.frame(ID=a0[1:alth,c(1:4)],LM=lmpre,GLM=glmpre,NB=NBpre,XG=XGpre,RF=RFpre)
         },
         "max" = {
           SMAX=data.frame(ID=a0[1:alth,c(1:4)],LM=lmpre,GLM=glmpre,NB=NBpre,XG=XGpre,RF=RFpre)
         },
         "min" = {
           SMIN=data.frame(ID=a0[1:alth,c(1:4)],LM=lmpre,GLM=glmpre,NB=NBpre,XG=XGpre,RF=RFpre)
         }
  )
}

#Report
Precis[Precis<0.2/tm]=0
Precis[,-1]=round(Precis[,-1],1)
Recall[Recall<2]=0
Recall[,-1]=round(Recall[,-1],0)
st=c(sum(is.na(Precis[,2:6])+is.na(Recall[,2:6]))+length(which(Precis==Inf))+length(which(Recall==Inf)), ST_cnt)
Precis;Recall;st

if (tm==2) {
  statis=read.table("AIHOT.txt", header = T);
  statis=rbind(statis,st)
  write.table(statis, "AIHOT.txt", quote = F, sep = "\t", row.names = F)
  par(mfrow=c(3,1))
  hist(stavg);plot(statis$AI);plot(statis$HOT)
}

RecF=data.frame(WAY=c("DIRE","BUY","SMAX","SMIN"),LM=0,NB=0,GLM=0,XG=0,RF=0)
PreF=RecF
for (i in 1:4) {
  for (i2 in 2:6) {
    if(is.na(Precis[i,i2])==0 && Precis[i,i2]!=Inf && Precis[i,i2]>1){
      PreF[i,i2]=1
    }
    if(is.na(Recall[i,i2])==0 && Recall[i,i2]!=Inf && Recall[i,i2]>10){
      RecF[i,i2]=1
    }
  }
}
PreF;RecF

Pre_interarr=function(data, nth){
  array_flag=which(PreF[nth,]==1)
  array_length=length(array_flag)
  if(array_length>0){
    for (i in array_flag) {
      temp=which(data[,i+3]==max(data[,i+3]))
      chose<<-intersect(temp,chose)
    }
  }
}
Rec_interarr=function(data, nth){
  array_flag=which(RecF[nth,]==1)
  array_length=length(array_flag)
  if(array_length>0){
    for (i in array_flag) {
      temp=which(data[,i+3]==max(data[,i+3]))
      chose<<-intersect(temp,chose)
    }
  }
}

chose=1:nrow(PDATA)
Rec_interarr(DIRE, 1)
Rec_interarr(BUY, 2)
Rec_interarr(SMAX, 3)
Rec_interarr(SMIN, 4)

PDATA = PDATA[chose,]
PDATA <- PDATA[order(PDATA$ID.代码), ]
write_xlsx(PDATA,path="F:/Desktop/Res.xlsx")
