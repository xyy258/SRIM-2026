# Use this to plot multiple values of r = N/f, and other parameters

r = nothing
profile = nothing

for p in [2,3]
global profile = p      # set bᵢ (0=linear, 1=nonlinear, 2=exponential, 3=linear+exp decay)
    for ratio in [31.6, 75.0]
        global r = ratio

        if profile == 0         # linear
            include("Ekman 3D.jl")
        elseif profile == 1     # nonlinear
            for value in [5, 10, 15]
                global T = value
                include("Ekman 3D.jl")
            end
        elseif profile == 2     # exponential with fixed buoyancy difference
            for factor in [0.2, 0.5, 1, 1.5]
                global efactor = factor
                include("Ekman 3D.jl")
            end
        elseif profile == 3     # linear + exponential decay
            for factor in [0.1, 0.2, 1, 1.5]
                global efactor = factor
                include("Ekman 3D.jl")
            end
        end

    end
end