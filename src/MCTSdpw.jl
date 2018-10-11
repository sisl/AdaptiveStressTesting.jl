module MCTSdpw

# This module implements Monte Carlo Tree Search with double progressive widening.
#Both the number of actions considered from a state and the number of transition
#states are restricted and increased throughout the simulation.
#Documentation for this algorithm can be found in
#'Adding double progressive widening to upper confidence trees to cope with
#uncertainty in planning problems' in European Workshop on Reinforcement Learning and
#'A comparison of Monte Carlo tree search and mathematical optimization for large scale
#dynamic resource allocation'

export DPWParams, DPWModel, DPW, selectAction, Depth
export MCTSTracker, get_actions, get_q_values

using ..MDP
import ..MDP: simulate
using Random, CPUTime

using ..BoundedPriorityQueues

const Depth = Int64

mutable struct DPWParams
    d::Depth                    # search depth
    ec::Float64                 # exploration constant- governs trade-off between 
                                #exploration and exploitation in MCTS
    n::Int64                    # number of iterations
    k::Float64                  # first constant controlling action generation
    alpha::Float64              # second constant controlling action generation
    clear_nodes::Bool           # clear all nodes before selecting next action
    maxtime_s::Float64          # maximum time to iterate, seconds
    rng_seed::UInt64            # random number generator

    top_k::Int64                #track the top k executions

    DPWParams() = new()
    function DPWParams(d::Depth, ec::Float64, n::Int64, k::Float64, alpha::Float64, 
        clear_nodes::Bool, maxtime_s::Float64, 
        rng_seed::UInt64, top_k::Int64=10) 
        new(d, ec, n, k, alpha, clear_nodes, maxtime_s, rng_seed, top_k)
    end
end

mutable struct DPWModel
    model::TransitionModel      # generative model
    getAction::Function         # returns action for rollout policy
    getNextAction::Function     # generates the next action when widening of the 
                                #action space is appropriate
end

mutable struct StateActionStateNode
    n::UInt64
    r::Float64
    StateActionStateNode() = new(0,0)
end

mutable struct StateActionNode
    s::Dict{State,StateActionStateNode}
    n::UInt64
    q::Float64
end
StateActionNode() = StateActionNode(Dict{State, StateActionStateNode}(), 0, 0)

mutable struct StateNode
    a::Dict{Action,StateActionNode}
    n::UInt64
end
StateNode() = StateNode(Dict{Action, StateActionNode}(), 0)

include("mctstracker.jl")

mutable struct DPW{A<:Action}
    s::Dict{State,StateNode}
    p::DPWParams
    f::DPWModel
    rng::AbstractRNG

    tracker::MCTSTracker{A}
    top_paths::BoundedPriorityQueue{MCTSTracker{A},Float64}
end

function DPW(p::DPWParams, f::DPWModel, ::Type{A}) where {A<:Action}
    s = Dict{State,StateNode}()
    rng = MersenneTwister(p.rng_seed)
    tracker = MCTSTracker{A}()
    top_paths = BoundedPriorityQueue{MCTSTracker{A},Float64}(p.top_k, 
        Base.Order.Forward) #keep highest
    dpw = DPW(s, p, f, rng, tracker, top_paths)
end

#backward-looking
function saveBackwardState(dpw::DPW, old_d::Dict{State,StateNode}, 
    new_d::Dict{State,StateNode}, s_current::State)
    !haskey(old_d, s_current) && return new_d
    s = s_current
    while s != nothing
        new_d[s] = old_d[s]
        s = s.parent 
    end
    new_d
end
#forward-looking
function saveForwardState(old_d::Dict{State,StateNode}, new_d::Dict{State,StateNode}, s::State)
    if !haskey(old_d,s)
        return new_d
    end 
    new_d[s] = old_d[s]
    for sa in values(old_d[s].a)
        for s1 in keys(sa.s)
            saveForwardState(old_d,new_d,s1)
        end
    end
    new_d
end
function saveState(dpw::DPW, old_d::Dict{State,StateNode}, s::State)
    new_d = Dict{State,StateNode}()
    saveBackwardState(dpw, old_d, new_d, s)
    saveForwardState(old_d, new_d, s)
    new_d
end
function trace_q_values(dpw::DPW, s_current::State)
    q_values = Float64[] 
    !haskey(dpw.s, s_current) && return q_values
    s = s_current
    while s.parent != nothing
        q = dpw.s[s.parent].a[s.action].q
        push!(q_values, q)
        s = s.parent 
    end
    reverse(q_values)
end

function selectAction(dpw::DPW, s::State; verbose::Bool=false)
    if dpw.p.clear_nodes
        #save s, its successors, and its ancestors
        new_dict = saveState(dpw, dpw.s, s)
        empty!(dpw.s) #cleanup
        dpw.s = new_dict
    end

    # This function calls simulate and chooses the approximate best action 
    #from the reward approximations
    d = dpw.p.d
    starttime_us = CPUtime_us()
    for i = 1:dpw.p.n
        R, actions = dpw.f.model.goToState(s)

        #init tracker
        empty!(dpw.tracker)
        append_actions!(dpw.tracker, actions)
        qvals = trace_q_values(dpw, s)
        append_q_values!(dpw.tracker, qvals)

        R += simulate(dpw, s, d, verbose=verbose)

        #process tracker
        combine_q_values!(dpw.tracker)
        enqueue!(dpw.top_paths, dpw.tracker, R; make_copy=true)

        if CPUtime_us() - starttime_us > dpw.p.maxtime_s * 1e6
            if verbose
                println("Iterations completed: $i")
            end
            break
        end
    end
    dpw.f.model.goToState(s) #leave the sim in current state
    println("Size of sdict: ", length(dpw.s))
    cS = dpw.s[s]
    A = collect(keys(cS.a)) # extract the actions taken in current state
    nA = length(A)
    Q = zeros(Reward,nA) # approximate value for each action
    for i = 1:nA
        Q[i] = cS.a[A[i]].q
    end

    @assert !isempty(Q) #something went wrong... 

    qmax, i = findmax(Q)
    A[i]::Action # choose action with highest approximate value
end

function simulate(dpw::DPW,s::State,d::Depth;verbose::Bool=false)
    # This function returns the reward for one iteration of MCTSdpw
    if d == 0 || dpw.f.model.isEndState(s)
        return 0.0::Reward
    end
    if !haskey(dpw.s,s) # if state is not yet explored, add it to the set of states, 
        #perform a rollout
        dpw.s[s] = StateNode()
        return rollout(dpw,s,d)::Reward
    end
    dpw.s[s].n += one(UInt64)
    if length(dpw.s[s].a) <= dpw.p.k*dpw.s[s].n^dpw.p.alpha # criterion for new action generation
        a = dpw.f.getNextAction(s,dpw.rng) # action generation step
        if !haskey(dpw.s[s].a,a) # make sure we haven't already tried this action
            dpw.s[s].a[a] = StateActionNode()
        end
    else # choose an action using UCT criterion
        cS = dpw.s[s] # save current state so we do not have to iterate through map many times
        A = collect(keys(cS.a)) # extract the actions taken in current state
        nA = length(A)
        UCT = zeros(Reward,nA)
        nS = cS.n
        for i = 1:nA
            cA = cS.a[A[i]] #current action
            @assert nS > 0
            @assert cA.n > 0
            UCT[i] = cA.q + dpw.p.ec*sqrt(log(nS)/cA.n)
        end
        a = A[argmax(UCT)] # choose action with highest UCT score
    end

    push_action!(dpw.tracker, a) #track actions
    qval = dpw.s[s].a[a].q
    push_q_value!(dpw.tracker, qval) #track q_values

    #state transition
    sp,r = dpw.f.model.getNextState(s,a,dpw.rng) # deterministic transition
    if !haskey(dpw.s[s].a[a].s,sp) # if transition state not yet exists, add it
        dpw.s[s].a[a].s[sp] = StateActionStateNode()
        dpw.s[s].a[a].s[sp].r = r
    else
        #already exists
        dpw.s[s].a[a].s[sp].n += one(UInt64)
    end

    q = r + simulate(dpw,sp,d-1)
    cA = dpw.s[s].a[a]
    cA.n += one(UInt64)
    cA.q += (q-cA.q)/cA.n
    dpw.s[s].a[a] = cA
    return q::Reward
end

function rollout(dpw::DPW,s::State,d::Depth)
    # Runs a rollout simulation using the default policy
    if d == 0 || dpw.f.model.isEndState(s)
        return 0.0::Reward
    else
        a = dpw.f.getAction(s,dpw.rng)

        push_action!(dpw.tracker, a) #track actions

        sp,r = dpw.f.model.getNextState(s,a,dpw.rng)
        qval = (r + rollout(dpw,sp,d-1))::Reward
        push_q_value2!(dpw.tracker, qval) #track q_values, reverse order, so track them separate
        qval
    end
end

end # module
