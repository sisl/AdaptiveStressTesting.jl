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

using MCTSdpw

type StressTestResults
    #vector of top k paths
    rewards::Vector{Float64}
    action_seqs::Vector{Vector{ASTAction}}
    q_values::Vector{Vector{Float64}}

    function StressTestResults(k::Int64)
        obj = new()
        obj.rewards = zeros(k) 
        obj.action_seqs = Array(Vector{ASTAction}, k)
        obj.q_values = Array(Vector{Float64}, k)
        obj
    end
end

uniform_getAction(ast::AdaptiveStressTest) = uniform_getAction(ast.rsg)

function uniform_getAction(rsg::RSG)
    policy(s::ASTState, rng::AbstractRNG) = random_action(rsg) #rng not used
    policy #function compatible with getAction() in MDP and MCTSdpw
end

#Starts MCTS
function stress_test(ast::AdaptiveStressTest, mcts_params::DPWParams; verbose::Bool=true)
    dpw_model = DPWModel(transition_model(ast), uniform_getAction(ast.rsg), 
        uniform_getAction(ast.rsg))
    dpw = DPW(mcts_params, dpw_model, ASTAction)
    (mcts_reward, action_seq) = simulate(dpw.f.model, dpw, 
        (x,y)->selectAction(x,y), verbose=verbose)

    results = StressTestResults(mcts_params.top_k)
    k = 1
    for (tr, r) in dpw.top_paths
        results.rewards[k] = r
        results.action_seqs[k] = get_actions(tr) 
        results.q_values[k] = get_q_values(tr)
        k += 1
    end

    #sanity check
    if mcts_reward >= results.rewards[1]
        warn("mcts_reward=$(mcts_reward), top reward=$(results.rewards[end])")
    end

    results
end

#experimental: try not stepping
function stress_test2(ast::AdaptiveStressTest, mcts_params::DPWParams; verbose::Bool=true)
    #save and overwrite
    clear_node_old = mcts_params.clear_nodes
    n_old = mcts_params.n
    mcts_params.clear_nodes = false
    mcts_params.n *= ast.params.max_steps 

    dpw_model = DPWModel(transition_model(ast), uniform_getAction(ast.rsg), 
        uniform_getAction(ast.rsg))

    dpw = DPW(mcts_params, dpw_model, ASTAction)

    s = dpw.f.model.getInitialState(dpw.rng)
    selectAction(dpw, s, verbose=verbose)

    results = StressTestResults(mcts_params.top_k)
    k = 1
    for (tr, r) in dpw.top_paths
        results.rewards[k] = r
        results.action_seqs[k] = get_actions(tr) 
        #@show length(results.action_seqs[k])
        results.q_values[k] = get_q_values(tr)
        #@show length(results.q_values[k])
        k += 1
    end

    #restore
    mcts_params.clear_nodes = clear_node_old
    mcts_params.n = n_old 

    results
end
