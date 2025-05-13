tm=2

if (tm==1) {
  #in day
  setwd('F:/Desktop/THS/data/test')
}else{
  #out day
  setwd('C:/Users/z30060762/Desktop/THS/etf2')
}
library(tidyverse)
library(openxlsx) 
library(stringr)
library(naivebayes)
library(randomForest)
library(xgboost)
library(Matrix)
library(writexl)
library(dplyr)

list_name = dir("./",pattern = "-2ETF.xlsx$") 
re = map(list_name, ~ read.xlsx(., startRow = 1, sheet=1)) #skip header rows

#select data
data=data.frame()
for (i in 1:8) {
  d=re[[i]]
  #clean
  d[,"现手"]=as.numeric(str_replace_all(d[,"现手"], "[↑↓-]", ""))
  d=d[,-c(13:18,30)]
  for (j in 3:ncol(d)) {
    if (sum(grepl("--", d[, j]))>0){
      d[, j]=as.numeric(str_replace_all(d[, j], "--", ""))
    }
  }
  d=transform(d,DATE=list_name[i])
  data=rbind(data,d)
}
data[,27]=str_replace(data[,27], "-2ETF.xlsx", "")


#calculus of continue
alth=265
lth=alth*8
日期=data[,27]
a0=cbind(日期,data[,-27])
a0[a0==0]=0.0000001
#移动均值栈
tavg=avg=tcost=tdeal=rep(0,alth*8);
stavg=rep(0,alth);
for (i in 1:8) {
  for (j in 1:alth) {
    prt=(i-1)*alth+j;
    if (i>1){
      tdeal[prt]=tdeal[prt-alth]+a0[prt,"总手"];
      tcost[prt]=tcost[prt-alth]+a0[prt,"总金额"]; 
    }else{
      tdeal[prt]=a0[prt,"总手"];
      tcost[prt]=a0[prt,"总金额"];       
    }
    avg[prt]=a0[prt,"总金额"]/a0[prt,"总手"];
    tavg[prt]=tcost[prt]/tdeal[prt];
    if(avg[prt]<tavg[prt]){
      stavg[j]=stavg[j]-1;
    }else{
      stavg[j]=stavg[j]+1;
    }
  }
}

x1=a0[1:(lth-alth),]
x2=a0[(alth+1):lth,]
xn=(x2[,-(1:4)]-x1[,-(1:4)])/x1[,-(1:4)]
xn1=xn[(alth+1):(lth-alth),]
xn2=a0[(2*alth+1):lth,-(1:4)]-a0[1:(lth-2*alth),-(1:4)]


if (tm==1) {
  #in day
  x3=a0[1:(lth-2*alth),]
  x4=a0[(2*alth+1):lth,]
  x5=x3[1:(4*alth),]
  x6=x3[(2*alth+1):(6*alth),]
  xn3=(x4[,-(1:4)]-x3[,-(1:4)])/x3[,-(1:4)]
  xn5=(x6[,-(1:4)]-x5[,-(1:4)])/x5[,-(1:4)]
  TIME=substring(日期,6)  #日期第6位
  #pan
  x.ipan=as.numeric(xn3$内盘<0.01*tm)
  l.ipan1=as.numeric(xn3$内盘>0.001*tm)
  l.ipan2=as.numeric(xn5$内盘>0.001*tm)
  ipan=rep(0,alth)
  for (i in 1:6) {
    ipan=ipan+x.ipan[((i-1)*alth+1):(i*alth)]
  }
  l.ipan=l.ipan1[(length(l.ipan1)-alth+1):length(l.ipan1)]+l.ipan2[(length(l.ipan2)-alth+1):length(l.ipan2)]
  
  x.opan=as.numeric(xn3$外盘<0.01*tm)
  l.opan1=as.numeric(xn3$外盘>0.001*tm)
  l.opan2=as.numeric(xn5$外盘>0.001*tm)
  opan=rep(0,alth)
  for (i in 1:6) {
    opan=opan+x.opan[((i-1)*alth+1):(i*alth)]
  }
  l.opan=l.opan1[(length(l.opan1)-alth+1):length(l.opan1)]+l.opan2[(length(l.opan2)-alth+1):length(l.opan2)]
}else{
  #out day
  TIME=as.numeric(substring(日期,3,4))  #日期第3、4位
  #pan
  x.ipan=as.numeric(xn$内盘<0.01*tm)
  l.ipan1=as.numeric(xn$内盘>0.001*tm)
  l.ipan2=as.numeric(xn2$内盘>0.001*tm)
  ipan=rep(0,alth)
  for (i in 4:7) {
    ipan=ipan+x.ipan[((i-1)*alth+1):(i*alth)]
  }
  l.ipan=l.ipan1[(length(l.ipan1)-alth+1):length(l.ipan1)]+l.ipan2[(length(l.ipan2)-alth+1):length(l.ipan2)]
  x.opan=as.numeric(xn$外盘<0.01*tm)
  l.opan1=as.numeric(xn$外盘>0.001*tm)
  l.opan2=as.numeric(xn2$外盘>0.001*tm)
  opan=rep(0,alth)
  for (i in 4:7) {
    opan=opan+x.opan[((i-1)*alth+1):(i*alth)]
  }
  l.opan=l.opan1[(length(l.opan1)-alth+1):length(l.opan1)]+l.opan2[(length(l.opan2)-alth+1):length(l.opan2)]
}
#mark table
Recall=data.frame(WAY=c("DIRE","BUY","SMAX","SMIN"),LM=NA,GLM=NA,NB=NA,XG=NA,RF=NA)
Precis=data.frame(WAY=c("DIRE","BUY","SMAX","SMIN"),LM=NA,GLM=NA,NB=NA,XG=NA,RF=NA)
