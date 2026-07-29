# Use this to plot multiple values of r = N/f

r = nothing
for ratio in [31.6, 75.0]
    global r = ratio
    for factor in [0.75, 1, 1.5]
        global efoldfactor = factor
        include("Ekman 3D.jl")
    end
end