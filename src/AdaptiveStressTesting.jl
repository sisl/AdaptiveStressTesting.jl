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

# Adaptive stress testing using Monte Carlo Tree Search
# Perform trajectory searches in a black box simulator to find events and maximize total probability
# Actions are the random seeds
#
#Author: Ritchie Lee

#for now... move this to submodule
include("MDP.jl")
include("MCTSdpw.jl")

module AdaptiveStressTesting

export AdaptiveStressTest, ASTParams, ASTState, ASTAction, transition_model, get_reward_default,
        random_action, get_action_sequence, reset_rsg!

using MDP
using RLESUtils, RNGWrapper
import Base: hash, isequal, ==

const DEFAULT_RSGLENGTH = 3
const G_RNG = MersenneTwister(0) #not used

mutable struct ASTParams
    max_steps::Int64 # safety for runaways in sim
    rsg_length::Int64 # dictates number of unique available random seeds
    init_seed::Int64 # initial value of seed on construct
    reset_seed::Union{Void,Int64} #reset to this seed value on initialize()
end
ASTParams() = ASTParams(0, DEFAULT_RSGLENGTH, 0, nothing)

mutable struct AdaptiveStressTest
    params::ASTParams
    sim
    sim_hash::UInt64 #keeps the sim in sync

    initialize::Function #initialize(sim)
    update::Function #update(sim)
    isterminal::Function #isterminal(sim)
    get_reward::Function #get_reward(prob, event, isterminal, dist, ast, sim)

    t_index::Int64 #starts at 1 and counts up in ints
    rsg::RSG #random seed generator
    initial_rsg::RSG #initial
    reset_rsg::Union{Void,RSG} #reset to this RSG

    transition_model::TransitionModel

    function AdaptiveStressTest(p::ASTParams, sim, initialize_fn::Function,
                              update_fn::Function, isterminal_fn::Function,
                              reward_fn::Function=get_reward_default)
        ast = new()
        ast.params = p
        ast.sim = sim
        ast.sim_hash = hash(0)
        ast.initialize = initialize_fn
        ast.update = update_fn
        ast.isterminal = isterminal_fn
        ast.get_reward = reward_fn
        ast.rsg = RSG(p.rsg_length, p.init_seed)
        ast.initial_rsg = deepcopy(ast.rsg)
        ast.reset_rsg = p.reset_seed != nothing ? RSG(p.rsg_length, p.reset_seed) : nothing
        ast.transition_model = transition_model(ast)
        ast
    end
end

mutable struct ASTAction <: Action
    rsg::RSG
end
ASTAction(len::Int64=DEFAULT_RSGLENGTH, seed::Int64=0) = ASTAction(RSG(len, seed))

mutable struct ASTState <: State
    t_index::Int64 #sanity check that at least the time corresponds
    hash::UInt64 #hash sim state to match with ASTState
    parent::Union{Void,ASTState} #parent state, root=nothing
    action::ASTAction #action taken from parent, root=0
end

function ASTState(t_index::Int64, parent::Union{Void,ASTState}, action::ASTAction)
    s = ASTState(t_index, 0, parent, action)
    s.hash = hash(s) #overwrites 0
    s
end

transition_model(ast::AdaptiveStressTest) = transition_model(ast, ast.sim)

function transition_model(ast::AdaptiveStressTest, ::Any)
    function get_initial_state(rng::AbstractRNG) #rng is unused
        ast.t_index = 1
        ast.initialize(ast.sim)
        if ast.reset_rsg != nothing #reset if specified
            ast.rsg = deepcopy(ast.reset_rsg)
        end
        s = ASTState(ast.t_index, nothing, ASTAction(deepcopy(ast.initial_rsg)))
        ast.sim_hash = s.hash
        s
    end

    function get_next_state(s0::ASTState, a0::ASTAction, rng::AbstractRNG) #rng is unused
        @assert ast.sim_hash == s0.hash
        ast.t_index += 1
        set_global(a0.rsg)
        #saving the entire state of the MersenneTwister would require 770 * 4 bytes.  Instead,
        # for now, just save seed. alternatively, seed can be an array of ints less than size 770
        # and the rest be generated using hash() would need to reach deep into components to use
        # an RNG that is passed around.  TODO: consider doing this

        prob, event, dist = ast.update(ast.sim)
        s1 = ASTState(ast.t_index, s0, a0)
        ast.sim_hash = s1.hash
        r = ast.get_reward(prob, event, ast.isterminal(ast.sim), dist, ast, ast.sim)
        (s1, r)
    end

    function isterminal(s::ASTState)
        @assert ast.sim_hash == s.hash
        ast.isterminal(ast.sim)
    end

    function go_to_state(target_state::ASTState)
        #Get to state s by traversing starting from initial state
        s = get_initial_state(G_RNG)
        actions = get_action_sequence(target_state)
        R = 0.0
        for a in actions
            s, r = get_next_state(s, a, G_RNG)
            R += r
        end
        @assert s == target_state
        R, actions
    end

    TransitionModel(get_initial_state, get_next_state, isterminal, ast.params.max_steps,
                         go_to_state)
end

function get_reward_default(prob::Float64, event::Bool, terminal::Bool, dist::Float64,
                            ast::AdaptiveStressTest, sim) #ast and sim not used in default
    r = log(prob)
    if event
        r += 0.0
    elseif terminal #incur distance cost only if !event && terminal
        r += -dist
    end
    r
end

function reset_rsg!(ast::AdaptiveStressTest)
    ast.rsg = deepcopy(ast.initial_rsg)
end

function random_action(rsg::RSG)
    next!(rsg)
    ASTAction(deepcopy(rsg))
end

function get_action_sequence(s::ASTState)
    actions = ASTAction[]
    #Tracing up the tree relieves us from storing the entire history of actions at each node
    while s.parent != nothing
        push!(actions, s.action)
        s = s.parent
    end
    reverse!(actions)
end

hash(a::ASTAction) = hash(a.rsg)
function hash(A::Vector{ASTAction}) 
    h = hash(A[1])
    for i = 2:length(A)
        h = hash(h, hash(A[i]))
    end
    h
end
function hash(s::ASTState)
    h = hash(s.t_index)
    h = hash(h, hash(s.parent == nothing ? nothing : s.parent.hash))
    h = hash(h, hash(s.action))
    h
end

==(w::ASTAction,v::ASTAction) = w.rsg == v.rsg
==(w::ASTState,v::ASTState) = hash(w) == hash(v)
isequal(w::ASTAction,v::ASTAction) = isequal(w.rsg,v.rsg)
isequal(w::ASTState,v::ASTState) = hash(w) == hash(v)

include("ASTSim.jl")
export sample, sample_timed, play_sequence, uniform_policy

include("AST_MCTS.jl") #mcts dpw
export uniform_getAction, DPWParams, stress_test, stress_test2, StressTestResults

include("AST_MC.jl") #mc best
export mc_best

include("dual_sim_mode.jl")
export DualSim, get_dualsim_reward_default

include("vis.jl")
export d3tree

end #module
