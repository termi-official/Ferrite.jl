using JuAFEM, SparseArrays


function solve()
function generate_nc_grid(C::Type{Quadrilateral})
    nodes = Array{Node{2,Float64},1}(undef, 8)
    nodes[1] = Node{2,Float64}(Vec(0.0,0.0))
    nodes[2] = Node{2,Float64}(Vec(1.0,0.0))
    nodes[3] = Node{2,Float64}(Vec(0.0,1.0))
    nodes[4] = Node{2,Float64}(Vec(0.5,1.0))
    nodes[5] = Node{2,Float64}(Vec(1.0,1.0))
    nodes[6] = Node{2,Float64}(Vec(0.0,2.0))
    nodes[7] = Node{2,Float64}(Vec(1.0,2.0))
    nodes[8] = Node{2,Float64}(Vec(0.5,2.0))

    cells = Array{Quadrilateral,1}(undef, 3)
    cells[1] = Quadrilateral((1,2,5,3))
    cells[2] = Quadrilateral((3,4,8,6))
    cells[3] = Quadrilateral((4,5,7,8))

    facesets = Dict("bottom" => Set([(1,1)]))
    return Grid(cells, nodes, facesets=facesets)
end

grid = generate_nc_grid(Quadrilateral);

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh);

K = create_sparsity_pattern(dh);

ch = ConstraintHandler(dh);

∂Ω = union(getfaceset.((grid, ), ["bottom"])...);

dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

close!(ch)
update!(ch, 0.0);

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

P = [1 0 0 0 0 0 0
     0 1 0 0 0 0 0
     0 0 1 0 0 0 0
     0 0 0 1 0 0 0
     0 0 0.5 0.5 0 0 0
     0 0 0 0 1 0 0
     0 0 0 0 0 1 0
     0 0 0 0 0 0 1]

Â, f̂ = doassemble(cellvalues, K, dh);
A = (P'*Â)*P
A = SparseMatrixCSC(A)
f = P'*f̂
apply!(A, f, ch)

u = A \ f;

vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, P*u)
end

return Â, f̂, A, f, u, P
end
