# Sets the save folder for new file, and root file name of data file

if profile == 0
    save_folder = "Ekman/3D Simulation/Linear/Animations/"
    root = @sprintf("Ekman/Data/0/r=%.1f/",r)
elseif profile == 1
    save_folder = @sprintf("Ekman/3D Simulation/Nonlinear/Animations/T=%.1f/",T)
    root = @sprintf("Ekman/Data/1/r=%.1f, T=%.1f/",r,T)
elseif profile == 2
    save_folder = @sprintf("Ekman/3D Simulation/Exponential with fixed Δb/Animations/L=%.1f/",Lᴰ)
    root = @sprintf("Ekman/Data/2/r=%.1f, L=%.1f/",r,Lᴰ)
elseif profile == 3
    save_folder = @sprintf("Ekman/3D Simulation/Linear + exponential decay/Animations/L=%.1f/",Lᴰ)
    root = @sprintf("Ekman/Data/3/r=%.1f, L=%.1f/",r,Lᴰ)
elseif profile == 4
    save_folder = @sprintf("Ekman/3D Simulation/Softplus/Animations/T=%.1f/",T)
    root = @sprintf("Ekman/Data/4/r=%.1f, T=%.1f/",r,T)
else
    save_folder = @sprintf("Ekman/3D Simulation/Animations/%.1f/",T)
    root = @sprintf("Ekman/Data/r=%.1f/",r)
end
mkpath(save_folder)