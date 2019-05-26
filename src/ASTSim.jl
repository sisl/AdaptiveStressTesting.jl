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

mutable struct SampleResults
    reward::Float64
    action_seq::Vector{ASTAction}
end

mutable struct ActionSequence{A <: Action}
    sequence::Vector{A}
    index::Int64
end
ActionSequence(action_seq::Vector{A}) where {A<:Action} = ActionSequence{A}(action_seq, 1)

function action_seq_policy(action_seq::ActionSequence)
    action = action_seq.sequence[action_seq.index]
    action_seq.index += 1
    return action
end

# Compatible with MDP / MCTSdpw
action_seq_policy(action_seq::ActionSequence, s::State) = action_seq_policy(action_seq)

#compatible with policy() in MDP
uniform_policy(ast::AdaptiveStressTest, s::ASTState) = uniform_policy(ast.rsg, s)
uniform_policy(rsg::RSG, s::ASTState) = random_action(rsg)

function sample(ast::AdaptiveStressTest; verbose::Bool=true)
    reward, actions = simulate(ast.transition_model, ast.rsg, uniform_policy, verbose=verbose)
    actions = convert(Vector{ASTAction}, actions) #from Vector{Action}
    SampleResults(reward, actions)
end

function sample(ast::AdaptiveStressTest, nsamples::Int64; print_rate::Int64=1)
    #Samples are varied since ast.rsg is not reset and sampling is done in series
    #Parallel version will need deterministic splitting of ast.rsg
    results = Vector{SampleResults}(undef,nsamples)
    for i = 1:nsamples
        if mod(i, print_rate) == 1
            println("sample ", i, " of ", nsamples)
        end
        results[i] = sample(ast, verbose=false) 
    end
    results #vector of tuples(reward, actions)
end

function sample_timed(ast::AdaptiveStressTest, maxtime_s::Float64; print_rate::Int64=1)
    #Samples are varied since ast.rsg is not reset and sampling is done in series
    tstart = CPUtime_start()
    results = Tuple{Float64,Vector{Action}}[]
    while true #do while structure guarantees at least 1 sample
        if mod(i, print_rate) == 1
            println("sample ", i, " of ", nsamples)
        end
        tup = sample(ast, verbose=false)
        push!(results, tup)
        if CPUtime_elapsed_s(tstart) > maxtime_s
            break
        end
    end
    results #nsamples = length(results)
end

function play_sequence(ast::AdaptiveStressTest, actions::Vector{A}; verbose::Bool=true) where {A<:Action}
    reward2, actions2 = simulate(ast.transition_model, ActionSequence(actions), 
        action_seq_policy, policy_length=length(actions), verbose=verbose)
    actions2 = convert(Vector{ASTAction}, actions2) #from Vector{Action}
    @assert actions == actions2 #check replay
    (reward2, actions2)
end

# this version returns the full rewards vector
#
function play_sequence_detailed(ast::AdaptiveStressTest, actions::Vector{A}; verbose::Bool=true) where {A<:Action}
    rewards2, actions2 = simulate_detailed(ast.transition_model, ActionSequence(actions), 
        action_seq_policy, policy_length=length(actions), verbose=verbose)
    actions2 = convert(Vector{ASTAction}, actions2) #from Vector{Action}
    @assert actions == actions2 #check replay
    (rewards2, actions2)
end

function test_sim(ast::AdaptiveStressTest, N::Int=1; verbose::Bool=false)
    #test N times
    for i = 1:N
        verbose && println("Testing $i of $N")
        #simulate
        rewards, actions = simulate_detailed(ast.transition_model, ast.rsg, uniform_policy, verbose=false)
        actions = convert(Vector{ASTAction}, actions) #from Vector{Action}
        verbose && println("simulation rewards: $rewards")
        verbose && println("simulation actions: $actions")
        #replay
        rewards2, actions2 = play_sequence_detailed(ast, actions; verbose=false) 
        verbose && println("replay rewards: $rewards2")
        verbose && println("replay actions: $actions2")
        if (rewards != rewards2) || (actions != actions2)
            return false #test failed
        end
    end
    return true #test passed
end
