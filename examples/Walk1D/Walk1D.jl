# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter Simulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter Simulator (RLES) includes the following
# third party software. The SISLES.jl package is licensed under the MIT Expat
# License: Copyright (c) 2014: Youngjun Kim.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
# "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# *****************************************************************************

"""
A 1D random walk that selects the next position according to a normal
distribution from the current position.  Under MC sampling, 
the walker has very low probability of exceeding the bounds within the given
time steps.  But under stress test, the bounds are exceeded in a small
number of steps.
"""
module Walk1D

using Distributions

export Walk1DParams, Walk1DSim, initialize, update, isterminal, isevent

mutable struct Walk1DParams
    startx::Float64
    threshx::Float64 #+- thresh
    endtime::Int64
    logging::Bool
end
Walk1DParams() = Walk1DParams(1.0, 10.0, 20, false)

mutable struct Walk1DSim
    p::Walk1DParams #parameters
    x::Float64
    t::Int64 #num steps
    distribution::Distribution
    log::Vector{Float64}
end

#Default to zero-mean Gaussian
function Walk1DSim(params::Walk1DParams, sigma::Float64)
    Walk1DSim(params, Normal(0.0, sigma))
end

#Option to set own distribution
function Walk1DSim(params::Walk1DParams, distribution::Distribution)
    Walk1DSim(params, params.startx, 0, distribution, Float64[])
end

function initialize(sim::Walk1DSim)
    sim.t = 0
    sim.x = sim.p.startx
    empty!(sim.log)
    if sim.p.logging
        push!(sim.log, sim.x)
    end
end

function update(sim::Walk1DSim)
    sim.t += 1
    r = rand(sim.distribution)
    sim.x += r
    prob = pdf(sim.distribution, r)
    dist = max(sim.p.threshx - abs(sim.x), 0.0) #non-negative
    if sim.p.logging
        push!(sim.log, sim.x)
    end
    return (prob, isevent(sim), dist)
end

function isevent(sim::Walk1DSim)
    abs(sim.x) >= sim.p.threshx #out-of-bounds in +-
end

function isterminal(sim::Walk1DSim)
    isevent(sim) || sim.t >= sim.p.endtime
end

end #module
