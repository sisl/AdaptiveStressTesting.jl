mutable struct MCTSTracker{A<:Action}
    actions::Vector{A}
    q_values::Vector{Float64}
    q_values2_rev::Vector{Float64} #these are stored in reverse, append to q_values

    MCTSTracker{A}() where {A<:Action} = new{A}(A[], Float64[], Float64[])
end

function Base.empty!(tr::MCTSTracker)
    empty!(tr.actions)
    empty!(tr.q_values)
    empty!(tr.q_values2_rev)
end

Base.hash(tr::MCTSTracker) = hash(tr.actions)
Base.:(==)(tr1::MCTSTracker, tr2::MCTSTracker) = tr1.actions == tr2.actions

push_action!(tr::MCTSTracker, a::Action) = push!(tr.actions, a)
push_q_value!(tr::MCTSTracker, q::Float64) = push!(tr.q_values, q)
push_q_value2!(tr::MCTSTracker, q2::Float64) = push!(tr.q_values2_rev, q2)

append_actions!(tr::MCTSTracker, a::AbstractVector{A}) where {A<:Action}= append!(tr.actions, a)
append_q_values!(tr::MCTSTracker, q::AbstractVector{Float64}) = append!(tr.q_values, q)

function combine_q_values!(tr::MCTSTracker) 
    if !isempty(tr.q_values2_rev)
        append!(tr.q_values, reverse(tr.q_values2_rev))
        empty!(tr.q_values2_rev)
    end
end

get_actions(tr::MCTSTracker) = tr.actions
get_q_values(tr::MCTSTracker) = tr.q_values

