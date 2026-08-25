#!/usr/bin/env julia
# Julia counterpart to tools/bench_solve_vs_jacobian.cpp -- same ergodic
# pose sweep as bench_ergodic.jl, but timing proximity_jacobian (solve +
# full derivative) instead of solve_socp alone, for a direct
# apples-to-apples comparison against DCOL++'s solve+derivatives timing
# with the geometric init now the library default (DEVIATIONS.md §1d).
#
# Unlike bench_ergodic.jl, problem matrices are NOT pre-built here:
# proximity_jacobian(p1,p2) builds them internally from p1.r/p1.q,
# p2.r/p2.q, so timing it directly (just updating p2.r/p2.q per pose) is
# the real, complete public-API call a Julia caller actually makes --
# matching bench_solve_vs_jacobian.cpp's own methodology on the C++ side
# (which times proximity()/proximityJacobian() end to end, including their
# internal problemMatrices/combineProblemMatrices).
#
# Usage: julia --project=<path to DifferentiableCollisions.jl> tools/bench_ergodic_jacobian.jl <PairName>

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

    # Poses pre-generated (cheap, not what's being measured), but the SOCP
    # problem matrices themselves are built INSIDE proximity_jacobian each
    # call -- see this file's header comment.
    poses = Vector{Tuple{SVector{3,Float64},SVector{4,Float64}}}(undef, N)
    for i in 1:N
        poses[i] = systematic_pose((i - 1) * dt, r_min, r_max)
    end

    GC.gc()

    for i in 1:10
        p2.r, p2.q = poses[i]
        DCD.proximity_jacobian(p1, p2; pdip_tol=1e-10)
    end

    durations_us = Float64[]
    sizehint!(durations_us, N)
    failed = 0

    for i in 1:N
        p2.r, p2.q = poses[i]
        t0 = time_ns()
        alpha, _, _ = DCD.proximity_jacobian(p1, p2; pdip_tol=1e-10)
        t1 = time_ns()
        if isnan(alpha)
            failed += 1
            continue
        end
        push!(durations_us, (t1 - t0) / 1000.0)
    end

    succ = length(durations_us)
    avg_us = succ > 0 ? sum(durations_us) / succ : 0.0
    sort!(durations_us)
    median_us = succ > 0 ? durations_us[div(succ, 2) + 1] : 0.0
    p25 = percentile_sorted(durations_us, 25.0)
    p75 = percentile_sorted(durations_us, 75.0)
    p95 = percentile_sorted(durations_us, 95.0)

    @printf("CASE %s | avg_us=%.4f | median_us=%.4f | p25_us=%.4f | p75_us=%.4f | p95_us=%.4f | success=%.2f%% (%d succ, %d failed)\n",
            name, avg_us, median_us, p25, p75, p95, 100.0*succ/N, succ, failed)
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

# Each pair its own fresh `julia` process -- same reasoning as
# bench_ergodic.jl (GC/allocator pressure from one pair's run otherwise
# bleeds into a later pair's timing in a shared process).
if length(ARGS) != 1 || !haskey(pairs, ARGS[1])
    println("Usage: julia bench_ergodic_jacobian.jl <PairName>  where PairName in: ", join(sort(collect(keys(pairs))), ", "))
    exit(1)
end
name = ARGS[1]
p1, p2 = pairs[name]()
run_case(name, p1, p2, N, T_MAX, R_MIN, R_MAX)
