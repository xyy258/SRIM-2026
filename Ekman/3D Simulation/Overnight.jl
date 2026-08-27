# Use this to plot multiple values of r = N/f, and other parameters

r = nothing
profile = nothing

file = "Ekman 3D.jl"

for p in [4]
global profile = p      # set bᵢ (0=linear, 1=nonlinear, 2=exponential, 3=linear+exp decay, 4=softplus)
    for ratio in [25,50]
        global r = ratio

        if profile == 0         # linear
            include(file)
        elseif profile == 1     # nonlinear
            for value in [5, 10, 15]
                global T = value
                include(file)
            end

        elseif profile == 2     # exponential with fixed buoyancy difference of N
            for value in [5, 10, 15, 20, 30, 40, 50]
                global Lᴰ = value
                include(file)
            end

        elseif profile == 3     # linear + exponential decay
            for value in [5, 10, 20, 40]
                global Lᴰ = value
                include(file)
            end

        elseif profile == 4     # softplus
            for value in [5,10,15,20,30,40,50]
                global T = value
                include(file)
            end
        end
    end
end