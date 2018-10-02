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

module RNGWrapper

export RSG, set_global, next, next!, set_from_seed! 

using Random
using Base.Iterators, IterTools

struct RSG #Seed generator
    state::Vector{UInt32}
end
function RSG(len::Int64=1, seed::Int64=0)
    return seed_to_state_itr(len, seed) |> collect |> RSG
end

set_from_seed!(rsg::RSG, len::Int64, seed::Int64) = copy!(rsg.state, seed_to_state_itr(len, seed))
seed_to_state_itr(len::Int64, seed::Int64) = take(iterated(hash_uint32, seed), len)

function next!(rsg::RSG)
    map!(hash_uint32, rsg.state, rsg.state)
    return rsg
end
function next(rsg0::RSG)
    rsg1 = deepcopy(rsg0)
    next!(rsg1)
    return rsg1
end

set_global(rsg::RSG) = set_gv_rng_state(rsg.state)
set_gv_rng_state(i::UInt32) = set_gv_rng_state([i])
set_gv_rng_state(a::Vector{UInt32}) = Random.seed!(a) #
#set_gv_rng_state(a::Vector{UInt32}) = Base.dSFMT.dsfmt_gv_init_by_array(a) #doesn't seem to work anymore
hash_uint32(x) = UInt32(hash(x) & 0x00000000FFFFFFFF) #take lower 32-bits

Base.length(rsg::RSG) = length(rsg.state)
Base.hash(rsg::RSG) = hash(rsg.state)
Base.:(==)(rsg1::RSG, rsg2::RSG) = rsg1.state == rsg2.state
Base.isequal(rsg1::RSG, rsg2::RSG) = rsg1 == rsg2

end #module
