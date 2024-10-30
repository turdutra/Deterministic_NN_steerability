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

function iterated_critical_radius(rho::AbstractMatrix,polytope_list::Vector{<:Vector{<:Vector{<:Real}}})
    for polytope in polytope_list
        inner_radius = critical_radius(rho,polytope)
        outer_radius = inner_radius/shrinking_factor(polytope)
        if outer_radius<1
            local_bool=false
            return local_bool, inner_radius, outer_radius
        elseif inner_radius>=1
            local_bool=true
            return local_bool, inner_radius, outer_radius
        end            
    end
    return missing, missing, missing
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
    # TODO:  FIX THE POLYTOPE ORTHER THING i think here i should invert the order of the concat
    current_polytope = vcat([full_polytope[i] for i in 1:length(current_solution) if current_solution[i] == 1],[full_polytope[end+1-i] for i in 1:length(current_solution) if current_solution[i] == 1])

    current_temp = initial_temp
    best_solution = current_solution

    # o_bool will signal if the objective function of the polytope is considered a success, either detecting steering or non steering
    current_value, current_o_bool = objective(current_polytope,rho)
    best_value = current_value
    best_o_bool = current_o_bool
    buffer_iteration=0
    i=0

    while i + buffer_iteration < max_iter 
        i += 1

        # Generate a new candidate solution
        new_solution = neighbor(current_solution, adjacency_list)
        new_polytope = vcat(
            [full_polytope[j] for j in 1:num_elements if new_solution[j] == 1],
            [full_polytope[end + 1 - j] for j in 1:num_elements if new_solution[j] == 1]
        )

        new_value, new_o_bool = objective(new_polytope, rho)
        delta = new_value - current_value

        # Metropolis criterion for acceptance
        if delta < 0 || rand() < exp(-delta / current_temp)
            current_solution = new_solution
            current_value = new_value
            current_polytope = new_polytope
            current_o_bool = new_o_bool
        end

        # Update the best solution found so far
        if current_value < best_value
            best_solution = copy(current_solution)
            best_value = current_value
            best_polytope = current_polytope
            best_o_bool = current_o_bool
        end

        # Decrease the temperature according to the cooling schedule
        current_temp *= cooling_rate

        # Check if current_value is below the threshold
        if best_o_bool && buffer_iteration ==0
            buffer_iteration = max_iter-i-50
            println("Threshold value reached R=$best_value at iteration $i.")
        end

        i+=1
    end

    println("Final iteration $max_iter, Temperature $current_temp, Best Value $best_value")
    if best_o_bool
        return best_solution, best_polytope 
    else
        return missing, missing
    end
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


function objective_steer(polytope,rho)
    #is steerable if R_out<1
    R=critical_radius(rho,polytope)
    if R==0 
        return 1000, false
    end
    s_factor=shrinking_factor(polytope)
    R_out=R/s_factor
    if R_out<1
        o_bool=true
    else
        o_bool=false
    end
    return R_out, o_bool
end

function objective_local(polytope,rho)
    #Is unsteerable if R>=1
    R=critical_radius(rho,polytope)
    if R==0 
        return 1000, false
    end

    if R >= 1
        o_bool = true
    else
        o_bool = false
    end

    return 2-R, o_bool
end

function OptimizePolytope(rho,polytope, adjacency_list,initial_temp,cooling_rate,max_iter)
    inner_radius = critical_radius(rho,polytope)
    outer_radius = inner_radius/shrinking_factor(polytope)
    if outer_radius<1
        local_bool=false
        best_solution, best_polytope=simulated_annealing(objective_steer, initial_temp, cooling_rate, max_iter,polytope, adjacency_list,rho)

    elseif inner_radius>=1
        local_bool=true
        best_solution, best_polytope=simulated_annealing(objective_local, initial_temp, cooling_rate, max_iter,polytope, adjacency_list,rho)

    else
        return nothing, nothing, missing, missing, missing
    end
    if best_solution === missing
        best_polytope = missing
        new_inner_radius = missing
        new_outer_radius = missing

    else
        new_inner_radius = critical_radius(rho,best_polytope)
        new_outer_radius = new_inner_radius/shrinking_factor(best_polytope)
    end


    return best_solution, best_polytope, local_bool,new_inner_radius,new_outer_radius
end

function angles_to_polytope(angles_list)
    N = length(angles_list)
    total_vectors = 2 * N  # Total number of vectors after adding opposites
    unit_vectors = Vector{Vector{Float64}}(undef, total_vectors)
    
    # First, convert angles to unit vectors and store them
    for i in 1:N
        phi, theta = angles_list[i]
        
        # Calculate the Cartesian coordinates
        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)
        
        # Store the unit vector at position i
        unit_vectors[i] = [x, y, z]
    end
    
    # Then, add the opposites of the vectors
    for i in 1:N
        # Get the original vector
        original_vector = unit_vectors[i]
        
        # Compute the opposite vector
        opposite_vector = -original_vector
        
        # Calculate the position for the opposite vector
        position = N - i + 1
        
        # Store the opposite vector at position N + position
        unit_vectors[N + position] = opposite_vector
    end
    
    return unit_vectors
end


# Function to normalize a vector
function normalize_tuple(v)
    norm_v = norm(v)
    return [v[1] / norm_v, v[2] / norm_v, v[3] / norm_v]
end



function generate_10_vertex_polytope()
    num_vertices = 5
    r = 1.0
    h = 0.5
    theta_lower = LinRange(0, 2 * π, num_vertices+1)[1:end-1]  # Equivalent to endpoint=False

    lower_pentagon = [[r * cos(θ), r * sin(θ), -h] for θ in theta_lower]
    upper_pentagon = reverse([-1*v for v in lower_pentagon])

    # Create a list of vectors (list of tuples)
    vertices = [lower_pentagon; upper_pentagon]
    # Normalize each tuple
    vertices_normalized = [normalize_tuple(v) for v in vertices]

    θ = rand(Uniform(0, 2 * π)) 

    rotated_vertices = apply_rotation(vertices_normalized,θ)
    return rotated_vertices
end

# Rodrigues' Rotation Formula
function rodrigues_rotation(v, k, θ)
    v_rot = v .* cos(θ) .+ cross(k, v) .* sin(θ) .+ k .* dot(k, v) .* (1 - cos(θ))
    return v_rot
end

# Apply random rotation to the vertices
function apply_rotation(vertices_or_vertex, θ)
    # Generate a random unit vector for the rotation axis
    random_axis = randn(3)
    random_axis /= norm(random_axis)  # Normalize the axis


    # Check if the input is a single vertex or a list of vertices
    if typeof(vertices_or_vertex[1]) <: AbstractFloat
        return rodrigues_rotation(vertices_or_vertex, random_axis, θ)  # Single vertex
    elseif typeof(vertices_or_vertex[1]) <: AbstractVector 
        return [rodrigues_rotation(v, random_axis, θ) for v in vertices_or_vertex]  # List of vertices
    else
        error("Input must be either a single vertex or a list of vertices")
    end
end

# TODO:  FIX THE POLYTOPE ORTHER THING
function neighborFreevec(current_solution, iteration ,n_step=0.05)
    new_solution=copy(current_solution)
    # Number of elements in the current solution
    N = Int(length(current_solution)/2)
    
    # Randomly pick an index to alter
    idx = rand(1:N)
    
    # Extract the vector to alter
    vector = current_solution[idx]
    op_vector = current_solution[end + 1 - idx]
    
    # Define the maximum possible change in angles based on temperature
    #rotation_angle = min(temperature * π/8,0.1*π)  # Adjust π to control the influence of temperature
    if iteration< 100
        rotation_angle=n_step*π*8
    else
        rotation_angle=n_step*π
    end
    
    new_vector = apply_rotation(vector,rotation_angle)
    new_op_vector = -1* new_vector

    new_solution[idx] = new_vector
    new_solution[end + 1 - idx] = new_op_vector
    return new_solution
end


# TODO: The stopping criteria should take into account if it has crossed the threshold but not end immediatly, let it improove a little bit more
function simulated_annealing_freevec(objective, initial_temp, cooling_rate, max_iter,rho,n_step)
    current_polytope = generate_10_vertex_polytope()

    current_temp = initial_temp
    best_polytope = current_polytope

    # o_bool will signal if the objective function of the polytope is considered a success, either detecting steering or non steering
    current_value, current_o_bool = objective(current_polytope,rho)
    best_value = current_value
    best_o_bool = current_o_bool
    
    buffer_iteration=0
    i=0

    while i + buffer_iteration < max_iter
        new_polytope = neighborFreevec(current_polytope,i,n_step)
        new_value, new_o_bool = objective(new_polytope, rho)
        delta = new_value - current_value

        # Metropolis criterion for acceptance
        if delta < 0 || rand() < exp(-delta / current_temp)
            current_value = new_value
            current_polytope=new_polytope
            current_o_bool = new_o_bool
        end

        # Update the best solution found so far
        if current_value < best_value
            best_value = current_value
            best_polytope=copy(current_polytope)
            best_o_bool = current_o_bool
        end

        # Decrease the temperature according to the cooling schedule
        current_temp *= cooling_rate

        
        if best_o_bool && buffer_iteration ==0
            buffer_iteration = max_iter-i-50
        end
        i+=1
    end
    println("Final iteration $i, Temperature $current_temp, Best Value $best_value, o_bool = $best_o_bool")
    if best_o_bool
        return best_polytope 
    else
        return missing
    end
end


#TODO: I should add a test with the big polytope to speed things Update
function OptimizePolytopeFreeVec(rho, polytope, initial_temp, cooling_rate, max_iter, n_step=0.05)

    inner_radius = critical_radius(rho,polytope)
    outer_radius = inner_radius/shrinking_factor(polytope)

    if outer_radius<1

        local_bool=false
        steerable_polytope=simulated_annealing_freevec(objective_steer, initial_temp, cooling_rate, max_iter,rho,n_step)
        new_inner_radius = missing
        new_outer_radius = missing
        if !ismissing(steerable_polytope)
            new_inner_radius = critical_radius(rho,steerable_polytope)
            new_outer_radius = inner_radius/shrinking_factor(steerable_polytope)
        end
        return steerable_polytope, local_bool, new_inner_radius, new_outer_radius

    elseif inner_radius>=1
         
        local_bool=true
        unsteerable_polytope=simulated_annealing_freevec(objective_local, initial_temp, cooling_rate, max_iter,rho,n_step)
        new_inner_radius = missing
        new_outer_radius = missing
        if !ismissing(unsteerable_polytope)
            new_inner_radius = critical_radius(rho,unsteerable_polytope)
            new_outer_radius = new_inner_radius/shrinking_factor(unsteerable_polytope)
        end
        return unsteerable_polytope, local_bool, new_inner_radius, new_outer_radius

    else
        steerable_polytope=simulated_annealing_freevec(objective_steer, initial_temp, cooling_rate, max_iter,rho,n_step)
        if !ismissing(steerable_polytope)
            local_bool = false
            new_inner_radius = critical_radius(rho,steerable_polytope)
            new_outer_radius = inner_radius/shrinking_factor(steerable_polytope)

            return steerable_polytope, local_bool, inner_radius, new_outer_radius
        end

        unsterrable_polytope=simulated_annealing_freevec(objective_local, initial_temp, cooling_rate, max_iter,rho,n_step)
        if !ismissing(unsterrable_polytope)
            local_bool = true
            new_inner_radius = critical_radius(rho,unsterrable_polytope)
            new_outer_radius = inner_radius/shrinking_factor(unsterrable_polytope)

            return unsterrable_polytope, local_bool, inner_radius, new_outer_radius
        end
        return missing, missing, missing, missing, missing
    end
end





function negativity_calc(rho)
    rho_pt = partial_transpose(rho, [2, 2], 2)
    eigenvalues = eigen(rho_pt).values
    negative_eigenvalues = [val for val in eigenvalues if val < 0]
    return sum(abs, negative_eigenvalues)
end

function concurrence_calc(rho)
    Y = [0 -im; im 0]
    rho_tilde = kron(Y, Y) * conj(rho) * kron(Y, Y)
    R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
    eigenvalues = sort(real(eigen(R).values), rev=true)
    return max(0, eigenvalues[1] - eigenvalues[2] - eigenvalues[3] - eigenvalues[4])
end


