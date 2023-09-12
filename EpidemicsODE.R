covid19_deaths_global <- read.csv("C:/Users/Oscar/Downloads/time_series_covid19_deaths_global.csv")
covid19_recovered_global <- read.csv("C:/Users/Oscar/Downloads/time_series_covid19_deaths_global.csv")
covid19_confirmed_global <- read.csv("C:/Users/Oscar/Downloads/time_series_covid19_deaths_global.csv")

country<-'Korea, South'
#country <- 'Italy'

covid19_deaths<-(time_series_covid19_deaths_global['Country.Region']==country)
covid19_deaths<-(time_series_covid19_deaths_global[covid19_deaths])[-1:-4]
covid19_deaths<-as.numeric(covid19_deaths)

covid19_recovered<-(time_series_covid19_recovered_global['Country.Region']==country)
covid19_recovered<-(time_series_covid19_recovered_global[covid19_recovered])[-1:-4]
covid19_recovered<-as.numeric(covid19_recovered)

covid19_confirmed<-(time_series_covid19_confirmed_global['Country.Region']==country)
covid19_confirmed<-(time_series_covid19_confirmed_global[covid19_confirmed])[-1:-4]
covid19_confirmed<-as.numeric(covid19_confirmed)


dcovid=data.frame(C=diff(covid19_confirmed), D=diff(covid19_deaths), R=diff(covid19_recovered))
acovid=data.frame(C=covid19_confirmed, D=covid19_deaths, R=covid19_recovered, I=covid19_confirmed-covid19_recovered-covid19_deaths)

write.csv(acovid[10:97,], file='acovid.csv')


tol<-1#1e-4
tol<-1/655000
acovid<-acovid*tol

XY<-acovid$I*(1.0-acovid$D-acovid$R-acovid$I)
dI<-diff(c(0,acovid$I))
dR<-diff(c(0,acovid$R))
dD<-diff(c(0,acovid$D))

model<-lm(c(dI,dR,dD)~c(XY,0*XY,0*XY)+c(-acovid$I, acovid$I,0*XY)+c(0*XY,-acovid$I,acovid$I)+0)
dModel<-matrix(predict(model), ncol=3)


cbind(cumsum(dModel[,1]),cumsum(dModel[,2]),cumsum(dModel[,3]))

julia> DD_res.minimizer
5-element Array{Float64,1}:
  30.50014979866272
5.311280018175142
0.0
0.07640386170586455
-0.1272379748989784