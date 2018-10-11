# *****************************************************************************
# Written by Ritchie Lee, ritchie.lee@sv.cmu.edu
# *****************************************************************************
# Copyright ã 2015, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.  The Reinforcement Learning Encounter sim1ulator (RLES)
# platform is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You
# may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
# law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# _____________________________________________________________________________
# Reinforcement Learning Encounter sim1ulator (RLES) includes the following
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

include("Walk1DDual.jl")
using Main.Walk1DDual
using AdaptiveStressTesting

const MAXTIME = 20 #sim endtime
const RNG_LENGTH = 2
const SIGMA = 1.0 #standard deviation of Gaussian
const SEED = 0 

sim1_params = SafeWalkParams()
sim1_params.startx = 1.0
sim1_params.threshx = 10.0
sim1_params.endtime = MAXTIME
sim1_params.logging = true
sim1_params.safe = (false, false) #one bound is safe
sim1 = SafeWalkSim(1, sim1_params, SIGMA)

sim2_params = deepcopy(sim1_params) 
sim2_params.safe = (true, false) #both bounds safe
sim2 = SafeWalkSim(2, sim2_params, SIGMA)

sim = DualSim(sim1, sim2)

ast_params = ASTParams(MAXTIME, RNG_LENGTH, SEED, nothing)
ast = AdaptiveStressTest(ast_params, sim, Walk1DDual.initialize, Walk1DDual.update, Walk1DDual.isterminal)

#For testing only
#sample(ast)

mcts_params = DPWParams()
mcts_params.d = 50
mcts_params.ec = 100
mcts_params.n = 200
mcts_params.k = 0.5
mcts_params.alpha = 0.85
mcts_params.clear_nodes = true
mcts_params.maxtime_s = floatmax(Float64)
mcts_params.rng_seed = UInt64(0)
mcts_params.top_k = 10

result = stress_test(ast, mcts_params)
reward, action_seq = result.rewards, result.action_seqs
