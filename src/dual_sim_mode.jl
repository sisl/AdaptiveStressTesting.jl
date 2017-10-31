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

#methods to support dual simulation mode (two simulations side by side)
#Author: Ritchie Lee

import Base.length

type DualSim
    sim1
    sim2
    get_reward::Function    

    function DualSim(sim1, sim2, get_reward::Function)
        ds = new(sim1, sim2, get_reward)
        finalizer(ds, x->begin
            finalize(x.sim1)
            finalize(x.sim2)
        end)
        ds
    end
end
DualSim(sim1, sim2) = DualSim(sim1, sim2, get_dualsim_reward_default)
length(dualsim::DualSim) = 2 

get_dualsim_reward_default(r1::Float64, r2::Float64) =  r1 - r2

function transition_model(ast::AdaptiveStressTest, ::DualSim)
    function get_initial_state(rng::AbstractRNG) #rng is unused
        ast.t_index = 1
        ds = ast.sim
        ast.initialize(ds.sim1)
        ast.initialize(ds.sim2)

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

        ds = ast.sim

        #saving the entire state of the MersenneTwister would require 770 * 4 bytes.  Instead,
        # for now, just save seed. alternatively, seed can be an array of ints less than size 770
        # and the rest be generated using hash() would need to reach deep into components to use
        # an RNG that is passed around.  TODO: consider doing this

        #sim1
        set_global(a0.rsg)
        prob, event, dist = ast.update(ds.sim1)
        r1 = ast.get_reward(prob, event, ast.isterminal(ds.sim1), dist, ast, ds.sim1)

        #sim2
        set_global(a0.rsg)
        prob, event, dist = ast.update(ds.sim2)
        r2 = ast.get_reward(prob, event, ast.isterminal(ds.sim2), dist, ast, ds.sim2)

        r = ds.get_reward(r1, r2)

        s1 = ASTState(ast.t_index, s0, a0)
        ast.sim_hash = s1.hash
        (s1, r)
    end

    function isterminal(s::ASTState)
        @assert ast.sim_hash == s.hash
        ds = ast.sim
        ast.isterminal(ds.sim1) || ast.isterminal(ds.sim2)
    end

    #unchanged from single
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

