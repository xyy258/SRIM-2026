# Use this to plot multiple values of r = N/f

r = nothing
for ratio in [31.6, 75.0]
    global r = ratio
    for value in [15, 20, 30]
        global efold = value
        include("Ekman 3D.jl")
    end
end