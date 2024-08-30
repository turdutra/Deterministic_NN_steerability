using Printf
import Polyhedra
import LinearAlgebra

using Random
using Distributions
using JuMP
using Mosek
using MosekTools
using LinearAlgebra



### Print floats with 4 decimal digits
Base.show(io::IO, f::Float64) = @printf(io, "%1.4f", f)
###


### Some useful objects
Pauli_matrix = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
Hadamard = [1 1; 1 -1]/sqrt(2)
phase_gate(phi) = [1 0; 0 exp(im*phi)]
###


### Useful functions for matrix manipulations
function index_to_array(k, dims) #= dims = [2, 2, 2] or something =#
    n_parties = length(dims)

    array = ones(n_parties)
    for i = 2:k
        array[n_parties] = array[n_parties] + 1
        for j in n_parties:-1:1
            if array[j] > dims[j]
                array[j-1] = array[j-1] + 1
                array[j] = 1
            end
        end
    end
    return array
end

function array_to_index(array, dims)
    n_parties = length(dims)
    index = 1
    for i = n_parties:-1:1
        prod = 1
        if i < n_parties
            for j = n_parties:(i+1)
                prod = prod*dims[j]
            end
        end
        index = index + (array[i] - 1)*prod
    end
    return Int64(index)
end

function partial_transpose(matrix, dims, axis) #= dims = [2, 2, 2] or something =#

    n_parties = length(dims)
    
    partially_transposed_matrix = copy(matrix)
    for i = 1:size(matrix)[1], j = 1:size(matrix)[2]
        
        array_i = index_to_array(i, dims)
        array_j = index_to_array(j, dims)
        
        new_array_i = copy(array_i)
        new_array_j = copy(array_j)
        
        new_array_i[axis] = array_j[axis]
        new_array_j[axis] = array_i[axis]

        new_index_i = array_to_index(new_array_i, dims)
        new_index_j = array_to_index(new_array_j, dims)

        
        partially_transposed_matrix[i, j] = matrix[new_index_i, new_index_j]
    end

    return partially_transposed_matrix
end



function partial_trace(matrix, dims, axis)
    
    n_parties = length(dims)


    new_dims = ones(n_parties - 1)
    for i = 1:n_parties
        if i < axis
            new_dims[i] = dims[i]
        elseif i > axis
            new_dims[i-1] = dims[i]
        end
    end

    matrix_dimension = 1
    for i=1:(n_parties-1)
        matrix_dimension = Int64(matrix_dimension*new_dims[i])
    end

    
    new_matrix = zeros(typeof(matrix[1,1]), matrix_dimension,matrix_dimension)
    for i = 1:size(matrix)[1], j = 1:size(matrix)[2]
        array_i = index_to_array(i, dims)
        array_j = index_to_array(j, dims)

        if array_i[axis] == array_j[axis]
            
            new_array_i = ones(n_parties - 1)
            new_array_j = ones(n_parties - 1)

            for k = 1:n_parties
                if k < axis
                    new_array_i[k] = array_i[k]
                    new_array_j[k] = array_j[k]
                elseif k > axis
                    new_array_i[k-1] = array_i[k]
                    new_array_j[k-1] = array_j[k]
                end
            end

            new_index_i = array_to_index(new_array_i, new_dims)
            new_index_j = array_to_index(new_array_j, new_dims)

            new_matrix[new_index_i, new_index_j] = new_matrix[new_index_i, new_index_j] + matrix[i, j]
            
            
        end
    end

    return new_matrix
end


###


### Useful functions when dealing with qubits
gate_hadamard() = [1 1; 1 -1]/sqrt(2)
gate_phase(phi) = [1 0; 0 exp(im*phi)]

#Computes the unitary corresponding to a rotation in the Bloch sphere (see https://en.wikipedia.org/wiki/Euler_angles and https://qubit.guide/2.12-composition-of-rotations)
function rotation2unitary(R::AbstractMatrix{<:Real})
    phi = acos(R[3,3])
    if phi == 0
        alpha = atan(-R[1,2], R[1,1])
        beta = 0
    else
        alpha = atan(R[1, 3], -R[2, 3])
        beta = atan(R[3, 1], R[3, 2])
    end
    return gate_phase(alpha)*gate_hadamard()*gate_phase(phi)*gate_hadamard()*gate_phase(beta)
end

#Writes a given state in its canonical form
function canonical_form!(rho::AbstractMatrix, mixBobsmarg::Bool = true)
    if mixBobsmarg
        map = sqrt(inv(partial_trace(rho, [2, 2], 1)))
        rho = kron(I(2), map)*rho*kron(I(2), map)
        parent(rho) ./= tr(rho)
    end
    T = real.([tr(rho*kron(Pauli_matrix[i], Pauli_matrix[j])) for i in 1:3, j in 1:3])
    
    (Diagonal(T) == T) && return rho
    
    U, _, V = svd(T)
    if det(U) < 0
        U = -U
    end
    if det(V) < 0
         V = -V
    end
    Ua = rotation2unitary(U')
    Ub = rotation2unitary(V')

    rho = kron(Ua, Ub)*rho*kron(Ua', Ub')
    return rho
end

function canonical_form(rho::AbstractMatrix, mixBobsmarg = true)
    return canonical_form!(copy(rho), mixBobsmarg)
end

function bloch_vec(A::AbstractMatrix)
    !(A ≈ A') && throw(ArgumentError("A is not hermitian, so it cannot be decomposed in the Gell-Mann basis"))
    !(tr(A) ≈ 1) && throw(ArgumentError("A does not have unit trace, consider using gm_vec instead"))
    d = size(A)[1]
    gm = gell_mann(d)
    return [Real(tr(gm[i]*A)) for i in 2:d^2]
end


#Tells whether a two-qubit quantum state is separable
function is_separable(rho)
    # Compute eigenvalues and eigenvectors
    rho_TA=partial_transpose(rho, [2, 2], 2)
    eigs = eigen(rho_TA)
    w = eigs.values

    # PPT Criterion: Are all eigenvalues >= 0?
    ppt = all(real(w) .>= 0)
    return ppt
end
###



# Função para calcular a negatividade
function negativity(rho)
    rho_pt = partial_transpose(rho, [2, 2], 2)
    eigenvalues = eigen(rho_pt).values
    negative_eigenvalues = [val for val in eigenvalues if val < 0]
    return sum(abs, negative_eigenvalues)
end

# Função para calcular a concorrência
function concurrence(rho)
    Y = [0 -im; im 0]
    rho_tilde = kron(Y, Y) * conj(rho) * kron(Y, Y)
    R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
    eigenvalues = sort(real(eigen(R).values), rev=true)
    return max(0, eigenvalues[1] - eigenvalues[2] - eigenvalues[3] - eigenvalues[4])
end



### Useful functions for manipulating polytopes

#For a given set of vertices describing a polytope, this function computes the polytope description in terms of inequalities
function vertices_to_facets(vertices)
    half_space_rep = Polyhedra.MixedMatHRep(Polyhedra.doubledescription(Polyhedra.vrep(vertices)))
    facet3D_vectors = [half_space_rep.A[i, 1:end] for i in 1:size(half_space_rep.A)[1]]
    offsets = half_space_rep.b
    return facet3D_vectors, offsets
end



#For a given set of vertices defining a polytope, this function computes the maximum radius of a inner sphere
function shrinking_factor(vertices)
    facet3D_vectors, offsets = vertices_to_facets(vertices)
    radius = minimum([abs(offsets[i])/LinearAlgebra.norm(facet3D_vectors[i]) for i in eachindex(offsets)])
    return radius
end



### Preliminary functions for Chau's method

#Given three points in 3d, returns the plane that passes through them
function plane(points::Vector{<:Vector{<:Real}})
    a = points[2] - points[1]
    b = points[3] - points[1]
    normal = cross(a, b)
    offset = dot(points[1], normal)
    return normal, offset
end

#Given a finite set of points, returns all triples composed of such points
function all_triples(points::AbstractVector{T}) where {T}
    n = length(points)
    triples = Vector{Vector{T}}(undef, binomial(n,3))
    count = 1
    for i=1:(n-2)
        for j in (i+1):(n-1)
            for k in (j+1):n
                triples[count] = [points[i], points[j], points[k]]
                count += 1
            end
        end
    end
    return triples
end


#Given a finite set of points in 3d, all_planes gets all the planes that pass through at least three of those points
function all_planes(points::Vector{Vector{T}}) where {T<:Real}
    triples = all_triples(points)
    normals = Vector{Vector{T}}(undef, length(triples))
    offsets = Vector{T}(undef, length(triples))
    for i in eachindex(triples)
        normals[i], offsets[i] = plane(triples[i]) 
    end
    return normals, offsets
end

#Chau's method
function critical_radius(rho::AbstractMatrix, polytope::Vector{<:Vector{<:Real}} )
    normals, offsets = all_planes(polytope)

    rho_canonical = canonical_form(rho)

    a = real.([tr(Pauli_matrix[i]*partial_trace(rho_canonical, [2, 2], 2)) for i in 1:3])
    T = real.([tr(rho_canonical*kron(Pauli_matrix[i], Pauli_matrix[j])) for i in 1:3, j in 1:3])

    model = Model(Mosek.Optimizer)
    set_silent(model)

    r = @variable(model)
    @variable(model, probs[eachindex(polytope)] .>= 0)

    b = [@expression(model, sum(probs[j]*abs(-offsets[i] + normals[i]'*polytope[j])/norm(-offsets[i]*a + T*normals[i]) for j in eachindex(polytope))) for i in eachindex(normals)]

    for i in eachindex(normals)
        @constraint(model, r <= b[i])
    end

    @constraint(model, sum(probs) == 1)
    @constraint(model, sum(probs[j]*polytope[j] for j in eachindex(polytope)) .== 0)

    @objective(model, Max, r)

    optimize!(model)

    return objective_value(model)
end




function G_matrix(n::Int, m::Int)
    """
    Generation of the random matrix from the Ginibre ensemble
    A complex matrix with elements having real and complex part 
    distributed with the normal distribution 
    
    input: dimensions of the Matrix G of size n x m (integers)
    output: array of matrix G of size n x m
    """
    real_part = randn(n, m)
    imag_part = randn(n, m)
    G = (real_part + im * imag_part) / sqrt(2)
    return G
end

function rho_Bures(n::Int)
    """
    Generation of a random mixed density matrix (Bures metric)
    Input: n = dimension of the density matrix (integer)
    Output: array of density matrix 
    """
    # Create random unitary matrix
    U, _ = qr(randn(n, n) + im * randn(n, n))
    U=Matrix(U)

    # Create random Ginibre matrix
    G = G_matrix(n, n)
    
    # Construct density matrix
    rho = (I(n) + U) * G * (G') * (I(n) + U')
    
    # Normalize density matrix
    rho = rho / tr(rho)
    return rho
end


function rho_HS(n::Int)
    """
    Generate a random mixed density matrix (Hilbert-Schmidt metric)
    Input: n = dimension of the density matrix (integer)
    Output: density matrix as a complex array
    """

    # Create a random Ginibre matrix using the pre-defined G_matrix function
    G = G_matrix(n, n)
    
    # Construct the density matrix
    rho = G * G'
    
    # Normalize the density matrix
    rho /= tr(rho)
    
    return rho
end



# Helper function to check if two vectors are approximately equal
approx_equal(v1, v2; atol=1e-6) = all(abs.(v1 .- v2) .< atol)
function order_polytope(polytope)
    n = length(polytope) ÷ 2
    unique_vectors = Vector{typeof(polytope[1])}()

    for vec in polytope
        if !any(approx_equal(-vec, v) for v in unique_vectors)
            push!(unique_vectors, vec)
        end
    end

    if length(unique_vectors) != n
        error("The input polytope does not have inversion symmetry.")
    end

    ordered_vectors = vcat(unique_vectors, -reverse(unique_vectors))

    return ordered_vectors
end

function simulated_annealing(objective, initial_temp, cooling_rate, max_iter,full_polytope, adjacency_list,rho)
    current_solution = zeros(Int, Int(length(full_polytope)/2))
    current_solution[randperm(Int(length(full_polytope)/2))[1:5]] .= 1
    current_temp = initial_temp
    best_solution = current_solution
    current_value = objective(current_solution,full_polytope,rho)
    best_value = current_value
    for i in 1:max_iter
        new_solution = neighbor(current_solution, adjacency_list)
        new_value = objective(new_solution,full_polytope,rho)
        delta = new_value - current_value

        # Metropolis criterion for acceptance
        if delta < 0 || rand() < exp(-delta / current_temp)
            current_solution = new_solution
            current_value = new_value
        end

        # Update the best solution found so far
        if current_value < best_value
            best_solution = copy(current_solution)
            best_value = current_value
        end

        # Decrease the temperature according to the cooling schedule
        current_temp *= cooling_rate

        
        if current_value<=10
            println("Final iteration $i, Temperature $current_temp, Best Value $best_value")
            return best_solution
        end
    end
    println("Final iteration $max_iter, Temperature $current_temp, Best Value $best_value")
    return missing
end


function neighbor(x, adjacency_list)
    ones_indices = findall(v -> Bool(v), x)
    zero_neighbors=[]
    idx1=1
    while isempty(zero_neighbors)
        # Get indexes of `1` values
        idx1 = rand(ones_indices)
        neighbors = adjacency_list[idx1]
        # Filter neighbors to include only those with x[idx] == 0
        zero_neighbors = filter(idx -> x[idx] == 0, neighbors)
    end
    idx0 = rand(zero_neighbors)
    x[idx1] = 0
    x[idx0] = 1
    # Flip a `0` to `1`

    return x
end


function objective_steer(x,full_polytope,rho)
    sub_polytope = vcat([full_polytope[i] for i in 1:length(x) if x[i] == 1],[full_polytope[end+1-i] for i in 1:length(x) if x[i] == 1])
    R=critical_radius(rho,sub_polytope)
    if R==0 || sum(x)<2
        return 1000*length(x)
    end
    R_out=R/shrinking_factor(sub_polytope)
    if R_out < 1
        return 2*sum(x)
    else
        return 2*length(x)+(R_out-1)
    end
end

function objective_local(x,full_polytope,rho)
    sub_polytope = vcat([full_polytope[i] for i in 1:length(x) if x[i] == 1],[full_polytope[end+1-i] for i in 1:length(x) if x[i] == 1])
    R=critical_radius(rho,sub_polytope)
    if R==0 || sum(x)<2
        return 1000*length(x)
    elseif R>=1
        return 2*sum(x)
    else
        return 2*length(x)+(1-R)
    end
end

function OptimizePolytope(rho,polytope, adjacency_list,initial_temp,cooling_rate,max_iter)
    inner_radius = critical_radius(rho,polytope)
    outer_radius = inner_radius/shrinking_factor(polytope)
    if outer_radius<1
        local_bool=false
        best_solution=simulated_annealing(objective_steer, initial_temp, cooling_rate, max_iter,polytope, adjacency_list,rho)

    elseif inner_radius>=1
        local_bool=true
        best_solution=simulated_annealing(objective_local, initial_temp, cooling_rate, max_iter,polytope, adjacency_list,rho)

    else
        return nothing, nothing, missing, inner_radius, outer_radius
    end
    if best_solution === missing
        best_polytope = missing
    else
        best_polytope = vcat([polytope[i] for i in 1:length(best_solution) if best_solution[i] == 1],[polytope[end+1-i] for i in 1:length(best_solution) if best_solution[i] == 1])
    end
    return best_solution, best_polytope, local_bool,inner_radius,outer_radius
end

