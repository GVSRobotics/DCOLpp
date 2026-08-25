#!/usr/bin/env julia
# Timing comparison: DCOL++ (C++) vs. the original DifferentiableCollisions.jl,
# for proximity_jacobian on the same shape pairs/poses used by
# gen_socp_reference.jl. See DEVIATIONS.md for the numbers this produced.
#
# Usage: julia --project=<path to DifferentiableCollisions.jl> tools/bench_socp.jl

using DifferentiableCollisions
using StaticArrays, LinearAlgebra, Random
const DCD = DifferentiableCollisions

Random.seed!(20260821)

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

pairs = [
    ("SphereSphere", () -> (DCD.Sphere(0.5), DCD.Sphere(0.3))),
    ("SphereCapsule", () -> (DCD.Sphere(0.4), DCD.Capsule(0.3, 1.2))),
    ("CapsuleCylinder", () -> (DCD.Capsule(0.3, 1.2), DCD.Cylinder(0.25, 0.9))),
    ("CylinderCone", () -> (DCD.Cylinder(0.25, 0.9), DCD.Cone(1.2, deg2rad(22)))),
    ("ConePolytope", () -> (DCD.Cone(1.2, deg2rad(22)), make_cube())),
    ("PolytopePolytope", () -> (make_cube(0.5), make_cube(0.35))),
    ("PolytopeEllipsoid", () -> (make_cube(), DCD.Ellipsoid(SMatrix{3,3}(Diagonal([1 / 0.6^2, 1 / 0.4^2, 1 / 0.5^2]))))),
    ("EllipsoidPolygon", () -> (DCD.Ellipsoid(SMatrix{3,3}(Diagonal([1 / 0.6^2, 1 / 0.4^2, 1 / 0.5^2]))), make_hex())),
    ("PolygonSphere", () -> (make_hex(), DCD.Sphere(0.35))),
]

const NITERS = 20_000

function bench_pair(name, ctor)
    p1, p2 = ctor()
    p1.r = 2 .* (@SVector randn(3)); p1.q = normalize(@SVector randn(4))
    p2.r = 2 .* (@SVector randn(3)); p2.q = normalize(@SVector randn(4))

    # warm up (JIT compile)
    for _ in 1:10
        DCD.proximity_jacobian(p1, p2; pdip_tol=1e-10)
    end

    t0 = time_ns()
    for _ in 1:NITERS
        DCD.proximity_jacobian(p1, p2; pdip_tol=1e-10)
    end
    t1 = time_ns()
    us_per_call = (t1 - t0) / NITERS / 1000.0
    println(name, ": ", round(us_per_call, digits=2), " us/call")
end

for (name, ctor) in pairs
    bench_pair(name, ctor)
end
