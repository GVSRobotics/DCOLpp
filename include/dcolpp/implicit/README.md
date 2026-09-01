# `dcolpp::implicit` — planned, not yet implemented

A second proximity engine for **smooth strictly-convex implicit shapes**
(superellipsoid, superelliptic cylinder, smoothed polytope / truncated cone,
sphere), from **iDCOL** (Anup Teejo Mathew et al., *Collision Detection with
Analytical Derivatives of Contact Kinematics*, RAL 2026).

The `socp` engine represents every primitive with linear orthant rows plus
second-order cones, so proximity is a fixed-size **SOCP**. `implicit` will
instead take each shape's implicit function `f(x) <= 0` and pose the
membership of the uniformly-scaled shape as a **non-linear orthant
condition** `f_i(x) <= alpha`, giving a small non-linear complementarity /
Newton-KKT system rather than a cone program. Analytical relative-SE(3)
derivatives (`NewtonResult::T`) reuse the `dcolpp::se3` layer shared with the
`socp` engine.

Vendored `core/` sources and build wiring will land here in a future
release; see NOTICE.md for attribution.
