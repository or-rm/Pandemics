using Plots
using DifferentialEquations
using CSV
using Flux
using Optim
using DiffEqFlux
using OrdinaryDiffEq
using DiffEqSensitivity

Th=1
Nn = 6e4
acovid = CSV.read("C:\\Users\\Oscar\\Documents\\acovid.csv", delim=',')
#acovid = CSV.read("C:\\Users\\Oscar\\Documents\\acovid.csv", DataFrame)
nObs=length(acovid.C)

covidData= [reshape(acovid.I, 1,nObs); reshape(acovid.R,1,nObs); reshape(acovid.D, 1, nObs)]/Nn

tdays=LinRange(0,Th,nObs)

function DSIRD(du, u, h, p, t)
    I,R,D = u
    α,β,δ,tau,K = p
    du[1] = dI = (α*I*(1.0-R-I-D) - β*I*h(p, t-tau)[1] - δ*I*h(p, t-tau)[1])
    du[2] = dR = β*I*h(p, t-tau)[1]#-γ*R
    du[3] = dD = δ*I*h(p, t-tau)[1]
end
p = [29.0,4.26, 0.106, 0.1,1]
u0= [11/Nn,0.0,0.0]
h(p,t) = (11/Nn)*ones(eltype(p), 20)
p = [22.0,4.26,1.116, 0.1,1]
prob = DDEProblem(DSIRD,u0,h,(0.0,Th),p)
sol= solve(prob, MethodOfSteps(Tsit5()))
plot(sol[1,:], label=["Infected" "Recovered" "Deaths"])
plot(sol, label=["Infected" "Recovered" "Deaths"])

function DD_predict_adjoint(p)
  α,β,δ,tau,K = p
  sol = concrete_solve(prob,MethodOfSteps(RK4()),u0*K,p,sensealg=TrackerAdjoint(),saveat=tdays, maxiters=1e5)
  return(Array(sol))
end

function DD_loss_adjoint(p)
  α,β,δ,tau,K = p
  prediction = DD_predict_adjoint(p)/K
  N = length(prediction[1,:])
  lossI= sum(abs, x for x in (covidData[1,1:N]-prediction[1,1:N]))/sum(covidData[1,1:N])
  lossR= sum(abs, x for x in (covidData[2,1:N]-prediction[2,1:N]))/sum(covidData[2,1:N])
  lossD= sum(abs, x for x in (covidData[3,1:N]-prediction[3,1:N]))/sum(covidData[3,1:N])
  loss = (lossI + lossR + lossD)+nObs*sum(abs, x for x in covidData[:,N+1:nObs])
  loss, prediction
end

cb = function (p,l,pred) #callback function to observe training
  α,β,δ,tau,K = p
  display(l)
  display(p)
  display(plot(solve(remake(prob, p=p, u0=u0*K), MethodOfSteps(RK4())), label=["Infectados" "Recuperados" "Muertos"]))
  display(plot!(tdays,covidData[1,:]*K, label="Infectados"))
  display(plot!(tdays,covidData[2,:]*K, label="Recuperados"))
  display(plot!(tdays,covidData[3,:]*K, label="Muertos"))
  return false # Tell it to not halt the optimization. If return true, then optimization stops
end
#p= [25.0,3.26, 0.10, 0.1,1]
cb(p,DD_loss_adjoint(p)...)

#DD_res = DiffEqFlux.sciml_train(DD_loss_adjoint, p, BFGS(initial_stepnorm=0.01), cb = cb)
p= [22.0,3.26, 0.106, 0.1,3]
u0 = [11, 0, 0] /Nn
prob = DDEProblem(DSIRD,u0*3,h,(0.0,Th),p)
cb(p,DD_loss_adjoint(p)...)
DD_res = DiffEqFlux.sciml_train(DD_loss_adjoint, p, LBFGS(), cb = cb)
p = DD_res.minimizer
DD_res = DiffEqFlux.sciml_train(DD_loss_adjoint, p, LBFGS(), cb = cb)
p = DD_res.minimizer
