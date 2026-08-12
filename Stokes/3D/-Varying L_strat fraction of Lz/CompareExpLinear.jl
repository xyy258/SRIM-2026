using Oceananigans, JLD2, Plots, Printf, Statistics

# Side-by-side comparison of the current exponential-background run against the
# archived linear-background run in "Centered - Linear/", for all three Ri.
#
#   julia --project=.. CompareExpLinear.jl
#
# Both runs share Re_s, domain, resolution and scheme (Centered advection); the
# only difference is the background stratification the buoyancy field is
# initialized with and relaxed toward:
#
#   linear:       N²_bg(z) = N∞²                  (the paper's set-up)
#   exponential:  N²_bg(z) = N∞²[1 − exp(−z/L)],  L = 10 δ_s
#
# Everything is normalized by the far-field N∞², so the axes mean the same thing
# in both runs. Three diagnostics per case:
#   (a) buoyancy gradient profile at the latest common time
#   (b) mixed-layer height h_m(t)
#   (c) integrated TKE(t)
#
# h_m is reported two ways, because the paper's literal definition is not
# neutral between the two backgrounds:
#   h_m       : first z where ∂⟨b⟩/∂z = 0.1 N∞²          (the paper's own)
#   h_m,local : first z where ∂⟨b⟩/∂z = 0.1 N²_bg(z)     (departure from the
#               profile the run actually started from)
# For the linear run the two coincide; for the exponential run the first starts
# at 1.05 δ_s at t = 0 by construction, so h_m,local is the honest measure of
# how much mixing has happened.

const ω  = 1.4075235e-4
const ν  = 1.0e-6
const U₀ = 0.015
const δ  = sqrt(2ν / ω)
const T_tide = 2π / ω
const L_strat = 10δ

N²_bg_norm_exp(z) = 1 - exp(-z / L_strat)
N²_bg_norm_lin(z) = 1.0

const here    = @__DIR__
const lin_dir = joinpath(here, "Centered - Linear")

cases = [("Ri0", 1.0 * ω^2), ("Ri500", 500.0 * ω^2), ("Ri2500", 2500.0 * ω^2)]

const col_exp = "#1F559B"
const col_lin = "#C46A1F"

default(fontfamily = "sans-serif", grid = true, gridalpha = 0.15,
        framestyle = :box, tickfontsize = 9, guidefontsize = 10,
        legendfontsize = 8, titlefontsize = 11)

# Lowest height where f rises through `level` (a vector, so the threshold may
# itself vary with z). Interpolated on g = f − level.
function first_crossing(z, f, level; zmin = -Inf)
    g = f .- level
    for i in 1:length(g)-1
        z[i] < zmin && continue
        if g[i] < 0 <= g[i+1]
            return z[i] - g[i] * (z[i+1] - z[i]) / (g[i+1] - g[i])
        end
    end
    return NaN
end

# Load one run and reduce it to time series + the profile at a requested time.
function load_run(dir, case, N²; bgnorm)
    fname = joinpath(dir, "output_" * case, "TidalBL3D_" * case * "_profiles.jld2")
    isfile(fname) || return nothing

    B  = FieldTimeSeries(fname, "B")
    U  = FieldTimeSeries(fname, "U")
    V  = FieldTimeSeries(fname, "V")
    uu = FieldTimeSeries(fname, "uu")
    vv = FieldTimeSeries(fname, "vv")
    ww = FieldTimeSeries(fname, "ww")

    times = B.times
    zc = znodes(B)
    zg = 0.5 .* (zc[1:end-1] .+ zc[2:end])
    dz = diff(zc)

    nt   = length(times)
    hm   = fill(NaN, nt)          # paper definition: 0.1 N∞²
    hml  = fill(NaN, nt)          # local definition: 0.1 N²_bg(z)
    hrig = fill(NaN, nt)          # Ri_g = 0.25
    tke  = zeros(nt)              # ∫ TKE dz / (U₀² δ_s)

    lvl_far   = fill(0.1, length(zg))
    lvl_local = 0.1 .* bgnorm.(zg)

    for n in 1:nt
        Bp = vec(interior(B[n]));  Up = vec(interior(U[n])); Vp = vec(interior(V[n]))
        G  = diff(Bp) ./ dz ./ N²
        S² = (diff(Up) ./ dz) .^ 2 .+ (diff(Vp) ./ dz) .^ 2

        hm[n]  = first_crossing(zg, G, lvl_far)
        hml[n] = first_crossing(zg, G, lvl_local)
        Rig    = (G .* N²) ./ max.(S², eps())
        hrig[n] = first_crossing(zg, Rig, fill(0.25, length(zg)); zmin = δ)

        # ⟨w²⟩ lives on the z-faces; move it to the centres before adding it to
        # the (centred) horizontal variances.
        wwf = vec(interior(ww[n]))
        wwc = length(wwf) == length(Up) + 1 ?
              0.5 .* (wwf[1:end-1] .+ wwf[2:end]) : wwf
        k = 0.5 .* (vec(interior(uu[n])) .- Up .^ 2 .+
                    vec(interior(vv[n])) .- Vp .^ 2 .+ wwc)
        kmid = 0.5 .* (k[1:end-1] .+ k[2:end])
        tke[n] = sum(kmid .* dz) / (U₀^2 * δ)
    end

    return (; times, zc, zg, dz, B, U, V, hm, hml, hrig, tke, N²)
end

# Gradient profile at the saved time closest to t_target.
function gradient_profile(run, t_target)
    n = argmin(abs.(run.times .- t_target))
    Bp = vec(interior(run.B[n]))
    return n, diff(Bp) ./ run.dz ./ run.N²
end

# Mean over the last whole tidal period, so the numbers are not a snapshot of
# the within-cycle modulation the paper describes in §4.2.
function cycle_mean(times, y, t_end)
    idx = findall(t -> t_end - T_tide <= t <= t_end, times)
    v = filter(!isnan, y[idx])
    isempty(v) ? NaN : mean(v)
end

summary_rows = []

for (case, N²) in cases
    exp_run = load_run(here,    case, N²; bgnorm = N²_bg_norm_exp)
    lin_run = load_run(lin_dir, case, N²; bgnorm = N²_bg_norm_lin)

    if exp_run === nothing || lin_run === nothing
        @warn "Missing profiles for $case — skipping"
        continue
    end

    # Compare at the latest whole tidal period both runs reached.
    n_common = Int(floor(min(exp_run.times[end], lin_run.times[end]) / T_tide))
    t_cmp = n_common * T_tide

    ne, Ge = gradient_profile(exp_run, t_cmp)
    nl, Gl = gradient_profile(lin_run, t_cmp)

    kg_e = findall(z -> z / δ <= 40, exp_run.zg)
    kg_l = findall(z -> z / δ <= 40, lin_run.zg)

    pa = plot(; ylabel = "z / δ_s", xlabel = "∂⟨b⟩ₓᵧ/∂z / N∞²",
              title = @sprintf("(a) gradient at ωt = %.1f (%d T)", ω * t_cmp, n_common),
              ylims = (0, 40), yticks = 0:5:40, xlims = (0, 1.5),
              legend = :topleft, foreground_color_legend = nothing,
              background_color_legend = nothing)
    plot!(pa, N²_bg_norm_exp.(exp_run.zg[kg_e]), exp_run.zg[kg_e] ./ δ;
          color = col_exp, linestyle = :dash, linewidth = 1.2, label = "exp background (t = 0)")
    plot!(pa, fill(1.0, length(kg_l)), lin_run.zg[kg_l] ./ δ;
          color = col_lin, linestyle = :dash, linewidth = 1.2, label = "linear background (t = 0)")
    plot!(pa, Ge[kg_e], exp_run.zg[kg_e] ./ δ; color = col_exp, linewidth = 2, label = "exponential")
    plot!(pa, Gl[kg_l], lin_run.zg[kg_l] ./ δ; color = col_lin, linewidth = 2, label = "linear")

    pb = plot(; ylabel = "h_m / δ_s", xlabel = "ωt",
              title = "(b) mixed-layer height", legend = :bottomright,
              foreground_color_legend = nothing, background_color_legend = nothing)
    plot!(pb, ω .* exp_run.times, exp_run.hm ./ δ; color = col_exp, linewidth = 1.6,
          label = "exp, 0.1 N∞²")
    plot!(pb, ω .* exp_run.times, exp_run.hml ./ δ; color = col_exp, linewidth = 1.4,
          linestyle = :dot, label = "exp, 0.1 N²_bg(z)")
    plot!(pb, ω .* lin_run.times, lin_run.hm ./ δ; color = col_lin, linewidth = 1.6,
          label = "linear")

    pc = plot(; ylabel = "∫TKE dz / (U₀² δ_s)", xlabel = "ωt",
              title = "(c) integrated TKE", legend = :bottomright,
              foreground_color_legend = nothing, background_color_legend = nothing)
    plot!(pc, ω .* exp_run.times, exp_run.tke; color = col_exp, linewidth = 1.4, label = "exponential")
    plot!(pc, ω .* lin_run.times, lin_run.tke; color = col_lin, linewidth = 1.4, label = "linear")

    Ri_label = case == "Ri0" ? "0 (passive scalar)" : case[3:end]
    fig = plot(pa, pb, pc; layout = (1, 3), size = (1500, 560),
               leftmargin = 6Plots.mm, bottommargin = 6Plots.mm,
               plot_title = "Ri = $Ri_label — exponential vs linear background",
               plot_titlefontsize = 13)
    savefig(fig, joinpath(here, "figures", "compare_exp_vs_linear_$case.png"))
    @info "Saved figures/compare_exp_vs_linear_$case.png"

    push!(summary_rows,
          (case = case, nT = n_common,
           hm_e   = cycle_mean(exp_run.times, exp_run.hm,  t_cmp) / δ,
           hml_e  = cycle_mean(exp_run.times, exp_run.hml, t_cmp) / δ,
           hm_l   = cycle_mean(lin_run.times, lin_run.hm,  t_cmp) / δ,
           rig_e  = cycle_mean(exp_run.times, exp_run.hrig, t_cmp) / δ,
           rig_l  = cycle_mean(lin_run.times, lin_run.hrig, t_cmp) / δ,
           tke_e  = cycle_mean(exp_run.times, exp_run.tke, t_cmp),
           tke_l  = cycle_mean(lin_run.times, lin_run.tke, t_cmp),
           tend_e = ω * exp_run.times[end], tend_l = ω * lin_run.times[end]))
end

println()
println("Cycle-averaged over the last common whole tidal period. Lengths in δ_s.")
@printf("%-8s %4s | %8s %8s %8s | %8s %8s | %9s %9s | %7s %7s\n",
        "case", "T", "hm_exp", "hm_loc", "hm_lin", "Rig_exp", "Rig_lin",
        "TKE_exp", "TKE_lin", "ωt_exp", "ωt_lin")
for r in summary_rows
    @printf("%-8s %4d | %8.2f %8.2f %8.2f | %8.2f %8.2f | %9.4f %9.4f | %7.1f %7.1f\n",
            r.case, r.nT, r.hm_e, r.hml_e, r.hm_l, r.rig_e, r.rig_l,
            r.tke_e, r.tke_l, r.tend_e, r.tend_l)
end
