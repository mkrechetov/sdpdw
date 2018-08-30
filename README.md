# sdpdw
SDP verification of D-Wave

File sdp-bm.jl solves SDP relaxation of BQP using non-convex Burer-Monteiro low-rank parametrization; the inner optimization step is a gradient descent with step equal 1/L.

File sdp-JuMP.jl solves SDP relaxation of BQP via Mosek solver.
