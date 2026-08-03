# Progress probe: prints model time of a running case every INTERVAL seconds.
# Reads the profiles JLD2 the simulation is writing; does not touch the run.
using JLD2, Printf

const T = 2π / 1.4075235e-4

function probe(case, interval)
    path = joinpath("output_" * case, "TidalBL3D_" * case * "_profiles.jld2")
    prev_t = NaN
    prev_w = NaN
    while true
        try
            t, it = jldopen(path) do f
                ks = keys(f["timeseries/t"])
                (f["timeseries/t/$(ks[end])"], parse(Int, ks[end]))
            end
            w = time()
            if !isnan(prev_t) && t > prev_t
                rate = (t - prev_t) / T / ((w - prev_w) / 3600)   # periods per wall hour
                @printf("PROBE %s  t=%.3f periods  iter=%d  rate=%.2f periods/hr  eta_15=%.1f hr\n",
                        case, t/T, it, rate, (15 - t/T) / rate)
            else
                @printf("PROBE %s  t=%.3f periods  iter=%d  (establishing rate)\n", case, t/T, it)
            end
            prev_t = t
            prev_w = w
        catch e
            @printf("PROBE %s  read failed: %s\n", case, sprint(showerror, e))
        end
        flush(stdout)
        sleep(interval)
    end
end

probe(ARGS[1], length(ARGS) > 1 ? parse(Int, ARGS[2]) : 300)
