ENV["GKSwstype"] = "100"

using Oceananigans, JLD2, Plots, Printf, LaTeXStrings
using Plots.PlotMeasures

# Import parameters
include("Parameters.jl")
# Sets the following:
# Root data file name:    "root"
# Folder to be saved in:  "save_folder"
include("Filename_anim.jl")


# ===================  #
## Buoyancy animation ##
# ===================  #

# Read in the first iteration to load grid
b_ic = FieldTimeSeries(root * "Buoyancy.jld2", "b")

## Load in coordinate arrays
xb, yb, zb = nodes(b_ic)

## Open data files
file_vel = jldopen(root * "Velocity.jld2")
file_b   = jldopen(root * "Buoyancy.jld2")

## Extract vector of iterations
iterations = parse.(Int, keys(file_vel["timeseries/t"]))
t_save = zeros(length(iterations))

@info "Making animation of buoyancy heatmaps..."

# Masking for (0, Lz)
Lzmask  = zb[findall(x -> x < Lz, zb)]
NLzmask = length(Lzmask)

# Masking for height from bottom of domain
z_mask = findall(x -> x < 60, zb)
zbmask = zb[z_mask]
Nzmask = length(zbmask)

bscale = (isnothing(r) || r == 0) ? 1 : N²

# Load initial buoyancy profile
b_initial = file_b["timeseries/b/$(iterations[1])"][:, 1, 1:NLzmask]

# Fixing colour limits for buoyancy difference plot
clim_abs = maximum(
    maximum(abs, (file_b["timeseries/b/$iter"][:, 1, 1:Nzmask] .- b_initial[:, 1:Nzmask])' / bscale)
    for iter in iterations
)
clim_val = max(1e-6, clim_abs * 1.05)

anim = @animate for (i, iter) in enumerate(iterations)
    if i % 200 == 0
        @info "Drawing frame $i / $(length(iterations))..."
    end

    b_xz = file_b["timeseries/b/$iter"][:, 1, 1:NLzmask]

    t = file_vel["timeseries/t/$iter"]
    t_save[i] = t

    b_max = maximum(b_xz' / bscale)
    b_xz_plot = heatmap(xb, Lzmask, b_xz' / bscale;
        color  = :thermal,
        clims  = (0, max(1e-6, 1.05 * b_max)),
        xlabel = L"x", ylabel = L"z",
        xlims  = (0, Lx), ylims = (0, Lz),
        dpi    = 300)

    b_diff_xz_plot = heatmap(xb, zbmask, (b_xz[:, 1:Nzmask] .- b_initial[:, 1:Nzmask])' / bscale;
        color  = :coolwarm,
        clims  = (-clim_val, clim_val),
        xlabel = L"x", ylabel = L"z",
        xlims  = (0, Lx), ylims = (0, zbmask[end]),
        dpi    = 300)

    t_round = round(Int, t)
    if isnothing(r) || r == 0
        b_title      = L"b\text{ at } t = %$(t_round)"
        b_diff_title = L"(b - b_i)\text{ at } t = %$(t_round)"
    else
        b_title      = L"b/N^2\text{ at } t = %$(t_round),\ N/f = %$(r)"
        b_diff_title = L"(b - b_i)/N^2\text{ at } t = %$(t_round),\ N/f = %$(r)"
    end

    # Combine sub-plots into a single figure
    plot(b_xz_plot, b_diff_xz_plot,
        layout = (2, 1),
        size   = (1000, 550),
        title  = [b_title b_diff_title],
        margin = 25px)
end
close(file_vel)
close(file_b)

# Save animation
mkpath(save_folder * "Buoyancy")
mp4(anim, save_folder * (isnothing(r) ? "Buoyancy/r_none.mp4" : @sprintf("Buoyancy/r = %.1f.mp4", r)), fps = 30)


#  ==========================  #
## Average velocity animation ##
#  ==========================  #

# Load FieldTimeSeries directly
u_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "u_avg")
v_avg_series = FieldTimeSeries(root * "Avg_vel.jld2", "v_avg")

times = u_avg_series.times
zu    = znodes(u_avg_series[1])

ylimits = (0, Lz)

@info "Making animation of plane-averaged velocity profiles..."

anim = @animate for i in 1:length(times)
    t = times[i]
    if i % 100 == 0
        @info "Drawing frame $i / $(length(times)) at sim time t = $(round(t, digits=1))..."
    end

    u_prof = vec(interior(u_avg_series[i], 1, 1, :))
    v_prof = vec(interior(v_avg_series[i], 1, 1, :))

    p1 = plot((u_prof .- U∞) / u_star, zu,
             linewidth = 3,
             color     = :navy,
             xlabel    = L"(\langle u \rangle - U_\infty)/u_*",
             ylabel    = L"z",
             xlims     = (-10, 7.5),
             ylims     = ylimits,
             grid      = true,
             margin    = 25px,
             legend    = false,
             dpi       = 300)

    p2 = plot(v_prof / u_star, zu,
             linewidth = 3,
             color     = :crimson,
             xlabel    = L"\langle v \rangle/u_*",
             ylabel    = L"z",
             xlims     = (-10, 7.5),
             ylims     = ylimits,
             grid      = true,
             margin    = 25px,
             legend    = false,
             dpi       = 300)

    title_str = isnothing(r) ? L"\text{Velocity profiles | } t = %$(round(t, digits=1))" : L"\text{Velocity profiles }(N/f = %$(r))\text{ | } t = %$(round(t, digits=1))"

    plot(p1, p2,
        layout     = (2, 1),
        size       = (1000, 600),
        margin     = 25px,
        plot_title = title_str,
        dpi        = 300)
end

mkpath(save_folder * "Velocity")
mp4(anim, save_folder * (isnothing(r) ? "Velocity/r_none.mp4" : @sprintf("Velocity/r = %.1f.mp4", r)), fps = 60)


# ============================= #
## Average vorticity animation ##
# ============================= #

@info "Loading vorticity time series..."
vort_file = root * "Avg_vort.jld2"

ωx_avg_series = FieldTimeSeries(vort_file, "ωx_avg")
ωy_avg_series = FieldTimeSeries(vort_file, "ωy_avg")
ωz_avg_series = FieldTimeSeries(vort_file, "ωz_avg")

vort_times = ωx_avg_series.times

zx = znodes(ωx_avg_series[1])
zy = znodes(ωy_avg_series[1])
zz = znodes(ωz_avg_series[1])

ylimits = (0, Lz)

@info "Making animation of plane-averaged vorticity profiles..."

anim_vort = @animate for i in 1:length(vort_times)
    t = vort_times[i]
    if i % 100 == 0
        @info "Drawing vorticity frame $i / $(length(vort_times)) at sim time t = $(round(t, digits=1))..."
    end

    ωx_prof = vec(interior(ωx_avg_series[i], 1, 1, :))
    ωy_prof = vec(interior(ωy_avg_series[i], 1, 1, :))
    ωz_prof = vec(interior(ωz_avg_series[i], 1, 1, :))

    p_x = plot(ωx_prof / f₀, zx,
               linewidth = 2,
               color     = :crimson,
               xlabel    = L"\langle \omega_x \rangle / f_0",
               ylabel    = L"z",
               xlims     = (-100, 100),
               ylims     = ylimits,
               grid      = true,
               legend    = false,
               dpi       = 300)

    p_y = plot(ωy_prof / f₀, zy,
               linewidth = 2,
               color     = :teal,
               xlabel    = L"\langle \omega_y \rangle / f_0",
               ylabel    = L"z",
               xlims     = (-50, 200),
               ylims     = ylimits,
               grid      = true,
               legend    = false,
               dpi       = 300)

    vort_title = isnothing(r) ? L"\text{Plane-Averaged Vorticity Profiles | } t = %$(round(t, digits=1))" : L"\text{Plane-Averaged Vorticity Profiles }(N/f = %$(r))\text{ | } t = %$(round(t, digits=1))"

    plot(p_x, p_y,
        layout     = (1, 2),
        size       = (1000, 600),
        margin     = 25px,
        plot_title = vort_title,
        dpi        = 300
    )
end

mkpath(save_folder * "Vorticity")
mp4(anim_vort, save_folder * (isnothing(r) ? "Vorticity/r_none.mp4" : @sprintf("Vorticity/r = %.1f.mp4", r)), fps = 60)