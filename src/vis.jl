
function d3tree(ast::AdaptiveStressTest, result::StressTestResults, get_text::Function;
               init_expand::Int=0)
    dpw = result.dpw
    S = dpw.s
    s = dpw.f.model.getInitialState(Base.GLOBAL_RNG) 

    children = Vector{Int}[] 
    text = String[]
    n = 0 

    stack = Tuple{Int,ASTState}[]
    push!(stack, (n+=1,s))
    while !isempty(stack)
        i,s = pop!(stack)
        R, actions = dpw.f.model.goToState(s)
        length(text) < i && resize!(text, i)
        text[i] = get_text(ast.sim)
        cs = get_children(dpw, s)
        cs = map(c->(n+=1,c), cs)
        length(children) < i && resize!(children, i)
        children[i] = [x[1] for x in cs]  
        push!(stack, cs...)
    end
    return D3Tree(children; text=text, init_expand=init_expand)
end

function get_children(dpw::DPW, s::ASTState)
    if haskey(dpw.s,s)
        [ASTState(s.t_index+1, s, a) for a in keys(dpw.s[s].a)]
    else
        ASTState[]
    end
end
