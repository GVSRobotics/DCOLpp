#!/usr/bin/env julia
# Julia counterpart to tools/bench_ergodic.cpp -- same ergodic pose sweep
# (ported from iDCOL's examples/ergodic.cpp), same statistics, for a direct
# apples-to-apples comparison against DCOL++'s solve-only timing.
#
# Usage: julia --project=<path to DifferentiableCollisions.jl> tools/bench_ergodic.jl

using DifferentiableCollisions
using StaticArrays, LinearAlgebra, Printf
const DCD = DifferentiableCollisions

function systematic_pose(t, r_min, r_max)
    f1, f2, f3 = sqrt(2.0), sqrt(3.0), sqrt(5.0)
    f4, f5, f6, f7 = sqrt(7.0), sqrt(11.0), sqrt(13.0), sqrt(17.0)
    TWO_PI = 2.0 * pi

    u = 0.5 * (1.0 + sin(TWO_PI * f1 * t))
    r_val = cbrt(r_min^3 + (r_max^3 - r_min^3) * u)
    theta = (pi / 2.0) * sin(TWO_PI * f2 * t)
    phi = TWO_PI * f3 * t

    pos = @SVector [r_val*cos(theta)*cos(phi), r_val*cos(theta)*sin(phi), r_val*sin(theta)]

    v1, v2 = sin(TWO_PI * f4 * t), cos(TWO_PI * f5 * t)
    v3, v4 = sin(TWO_PI * f6 * t), cos(TWO_PI * f7 * t)
    # DCD's q is scalar-first [w,x,y,z] (dcm_from_q: q4,q1,q2,q3 = q, i.e.
    # q[1]=w), matching Eigen::Quaterniond(v1,v2,v3,v4)'s (w,x,y,z) directly.
    q = normalize(@SVector [v1, v2, v3, v4])
    return pos, q
end

function percentile_sorted(v::Vector{Float64}, p::Float64)
    isempty(v) && return 0.0
    idx = Int(ceil(p / 100.0 * length(v)))
    idx = min(idx, length(v))
    idx = max(idx, 1)
    return v[idx]
end

function run_case(name, p1, p2, N, t_max, r_min, r_max)
    dt = t_max / N
    p1.r = @SVector zeros(3)
    p1.q = @SVector [1.0, 0.0, 0.0, 0.0]

    # Pre-generate all N (c,G,h,...) problems from the ergodic pose sweep,
    # untimed -- isolates solve_socp itself from problem-matrix construction
    # (matching bench_ergodic.cpp), while still sweeping 100k diverse poses.
    # Concretely typed (not Vector{Any}): boxing would add per-call runtime
    # dispatch INSIDE the timed loop below, unfairly inflating this side.
    G_ort1, h_ort1, G_soc1, h_soc1 = DCD.problem_matrices(p1, p1.r, p1.q)
    pos0, q0 = systematic_pose(0.0, r_min, r_max)
    p2.r = pos0; p2.q = q0
    G_ort2_0, h_ort2_0, G_soc2_0, h_soc2_0 = DCD.problem_matrices(p2, p2.r, p2.q)
    prob0 = DCD.combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1, G_ort2_0, h_ort2_0, G_soc2_0, h_soc2_0)
    problems = Vector{typeof(prob0)}(undef, N)
    problems[1] = prob0
    for i in 1:(N-1)
        t = i * dt
        pos, q = systematic_pose(t, r_min, r_max)
        p2.r = pos; p2.q = q
        G_ort2, h_ort2, G_soc2, h_soc2 = DCD.problem_matrices(p2, p2.r, p2.q)
        problems[i+1] = DCD.combine_problem_matrices(G_ort1, h_ort1, G_soc1, h_soc1, G_ort2, h_ort2, G_soc2, h_soc2)
    end

    durations_us = Float64[]
    sizehint!(durations_us, N)
    iters_used = Int[]
    sizehint!(iters_used, N)
    failed = 0

    # Running all 9 pairs back-to-back in one process otherwise lets GC
    # pressure from earlier pairs' 100k-problem arrays land mid-timing for
    # a later pair (a multi-hundred-us pause landing on one sample blows up
    # both the mean and stddev) -- C++ has no analogous issue (deterministic
    # destructors), so this would unfairly penalize Julia in a multi-pair
    # run. Force a clean GC state right before each pair's timed region.
    GC.gc()

    for i in 1:10
        c,G,h,idx_ort,idx_soc1,idx_soc2 = problems[i]
        DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; pdip_tol=1e-10)
    end

    for i in 1:N
        c,G,h,idx_ort,idx_soc1,idx_soc2 = problems[i]

        t0 = time_ns()
        result = DCD.solve_socp(c,G,h,idx_ort,idx_soc1,idx_soc2; pdip_tol=1e-10)
        t1 = time_ns()

        if length(result) == 4
            x,s,z,iters = result
        else
            failed += 1
            continue
        end
        push!(durations_us, (t1 - t0) / 1000.0)
        push!(iters_used, iters)
    end

    succ = length(durations_us)
    avg_us = succ > 0 ? sum(durations_us) / succ : 0.0
    sort!(durations_us)
    median_us = succ > 0 ? durations_us[div(succ, 2) + 1] : 0.0
    p25 = percentile_sorted(durations_us, 25.0)
    p75 = percentile_sorted(durations_us, 75.0)
    p95 = percentile_sorted(durations_us, 95.0)
    stddev = succ > 0 ? sqrt(sum((v - avg_us)^2 for v in durations_us) / succ) : 0.0
    avg_iters = succ > 0 ? sum(iters_used) / succ : 0.0

    @printf("CASE %s | avg_us=%.4f | median_us=%.4f | p25_us=%.4f | p75_us=%.4f | p95_us=%.4f | stddev_us=%.4f | avg_iters=%.3f | success=%.2f%% (%d succ, %d failed)\n",
            name, avg_us, median_us, p25, p75, p95, stddev, avg_iters, 100.0*succ/N, succ, failed)
end

function make_cube(half_side=0.5)
    A = SMatrix{6,3}([1.0 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1])
    b = SVector{6}(fill(half_side, 6))
    DCD.Polytope(A, b)
end
function make_hex()
    N = 6
    ns = [[cos(th); sin(th)] for th in 0:(2 * pi / N):(2 * pi * (N - 1) / N)]
    A = SMatrix{N,2}(vcat(transpose.(ns)...))
    b = SVector{N}(fill(0.4, N))
    DCD.Polygon(A, b, 0.1)
end
make_ellipsoid() = DCD.Ellipsoid(SMatrix{3,3}(Diagonal([1 / 0.6^2, 1 / 0.4^2, 1 / 0.5^2])))

const N = 100_000
const T_MAX = 100.0
const R_MIN, R_MAX = 0.05, 2.0

pairs = Dict(
    "SphereSphere" => () -> (DCD.Sphere(0.5), DCD.Sphere(0.3)),
    "SphereCapsule" => () -> (DCD.Sphere(0.4), DCD.Capsule(0.3, 1.2)),
    "CapsuleCylinder" => () -> (DCD.Capsule(0.3, 1.2), DCD.Cylinder(0.25, 0.9)),
    "CylinderCone" => () -> (DCD.Cylinder(0.25, 0.9), DCD.Cone(1.2, deg2rad(22))),
    "ConePolytope" => () -> (DCD.Cone(1.2, deg2rad(22)), make_cube()),
    "PolytopePolytope" => () -> (make_cube(0.5), make_cube(0.35)),
    "PolytopeEllipsoid" => () -> (make_cube(), make_ellipsoid()),
    "EllipsoidPolygon" => () -> (make_ellipsoid(), make_hex()),
    "PolygonSphere" => () -> (make_hex(), DCD.Sphere(0.35)),
)

# Each pair runs in its own fresh `julia` process (see run_all_ergodic.sh) --
# running all 9 back-to-back in one process left GC/allocator pressure from
# earlier pairs' 100k-problem arrays landing mid-timing for later pairs
# (confirmed: even an explicit GC.gc() before each pair's timed region did
# not fix it), which C++ has no analogous issue with (deterministic
# destructors) and would have unfairly penalized Julia's numbers here.
if length(ARGS) != 1 || !haskey(pairs, ARGS[1])
    println("Usage: julia bench_ergodic.jl <PairName>  where PairName in: ", join(sort(collect(keys(pairs))), ", "))
    exit(1)
end
name = ARGS[1]
p1, p2 = pairs[name]()
run_case(name, p1, p2, N, T_MAX, R_MIN, R_MAX)
