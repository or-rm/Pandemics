using Plots
using DifferentialEquations

function lotka_volterra(du,u,p,t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = α*x - β*x*y
        du[2] = dy = -δ*y + γ*x*y
end

u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]

prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5())

using Plots

function predict_adjoint(p) # Our 1-layer neural network
         Array(concrete_solve(prob,Tsit5(),u0,p,saveat=0.0:0.1:10.0))
end

function loss_adjoint(p)
         prediction = predict_adjoint(p)
         loss = sum(abs2,x-1 for x in prediction)
         loss,prediction
end

cb = function (p,l,pred) #callback function to observe training
         display(l)
         # using `remake` to re-create our `prob` with current parameters `p`
         display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.0:0.1:10.0),ylim=(0,6)))
         return false # Tell it to not halt the optimization. If return true, then optimization stops
end

res = DiffEqFlux.sciml_train(loss_adjoint, p, BFGS(initial_stepnorm = 0.0001), cb = cb)
