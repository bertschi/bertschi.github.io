# Some examples of configurable functions in Julia

struct ShapeInfo
    dims::Tuple{Vararg{Int64}}
end

function Base.:*(x::ShapeInfo, y::ShapeInfo)
    @assert x.dims[end] == y.dims[1]
    ShapeInfo((x.dims[1:end-1]..., y.dims[2:end]...))
end

struct NN{U,V}
    W::U
    b::V
end

Base.broadcasted(op, x::ShapeInfo) = x

function Base.broadcasted(op, x::ShapeInfo, y::ShapeInfo)
    n = max(length(x.dims), length(y.dims))
    dx = ntuple(i -> if i > length(x.dims) 1 else x.dims[i] end, n)
    dy = ntuple(i -> if i > length(y.dims) 1 else y.dims[i] end, n)
    ShapeInfo(tuple((max(i, j)
                     for (i, j) in zip(dx, dy)
                     if i == j || min(i, j) == 1 || error("Nonmatching dimensions ", i, " and ", j))...))
end

(nn::NN)(x) = tanh.(nn.W * x .+ nn.b)

# Code from markdown script

function fun(g, x, y)
    g(z * (x + y))
end

function fun2(g, x, y, z, plus, mul)
	g(mul(z, plus(x, y)))
end

@show fun2(sqrt, 1, 2, 3, +, *)

using Flux

@show fun2(Flux.relu, [1, 2], [1 2; 3 4] \ [-1, 2], [1 2; 3 4], .+ , *)

function rplus(r1, r2)
	Regex(string("(?:", r1.pattern, ")|(?:", r2.pattern, ")"))
end

@show fun2(r -> match(r, "cb"), r"a", r"b", r"c", rplus, *)

function fun(g, x, y, z)
    g(z*(x + y))
end

@show fun(Flux.relu, [1, 2], [1 2; 3 4] \ [-1, 2], [1 2; 3 4])

@show fun(sqrt, 1, 2, 3)

Base.:+(r1::Regex, r2::Regex) = rplus(r1, r2)

@show fun(identity, r"a", r"b", r"c")

using Symbolics

@variables x, y

@show fun(sqrt, x, y, 3)

fif(x, y, z) = if x y() else z() end
fif(x::Num, y, z) = :(if $x; $(y()) else $(z()) end)

recur(f, x) = f(x)
@register_symbolic recur(it)
recur(f, x::Num) = recur(x)

fac(n) = fif(n < 1, () -> 1, () -> n * recur(fac, n-1))
