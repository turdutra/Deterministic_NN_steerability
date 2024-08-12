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
Pauli_matrices = [[0 1; 1 0], [0 -im; im 0], [1 0; 0 -1]]
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

#To specify a projective measurement for a qubit, it is only necessary to specify a bloch vector. This function transforms the bloch vector in the actual projector.
function vector_to_hermitian(measurement_vectors::AbstractMatrix)
    N = size(measurement_vectors)[1]
    vectors_dot_sigma = [sum(measurement_vectors[i,j]*Pauli_matrices[j] for j in 1:3) for i in 1:N]
    projectors = [(LinearAlgebra.I(2) + vectors_dot_sigma[i])/2 for i in 1:N]
    return projectors
end

function vector_to_hermitian(measurement_vectors::AbstractVector)
    N = length(measurement_vectors)
    vectors_dot_sigma = [sum(measurement_vectors[i][j]*Pauli_matrices[j] for j in 1:3) for i in 1:N]
    projectors = [(LinearAlgebra.I(2) + vectors_dot_sigma[i])/2 for i in 1:N]
    return projectors
end

#Transforms a bloch vector into its corresponding ket state (in the computational basis)
function bloch_vector_to_ket(vector)
    theta = acos(vector[3])
    phi = atan(vector[2], vector[1])
    return [cos(theta/2), exp(im*phi)*sin(theta/2)]
end

#To specify a dichotomic measurement, it is only necessary to specify one of the measurement operators. The following function constructs the whole measurements
function all_measurement_elements(finite_measurements)
    n_measurements = size(finite_measurements)[1]
    measurement_elements = [finite_measurements[1]]
	for i=2:2*n_measurements
		if iseven(i)
			push!(measurement_elements, LinearAlgebra.I(2)- measurement_elements[i-1])
		else
			push!(measurement_elements, finite_measurements[1+floor(Int64, i/2)])
		end
	end
    return measurement_elements
end

#Computes the unitary corresponding to a rotation in the Bloch sphere (see https://en.wikipedia.org/wiki/Euler_angles and https://qubit.guide/2.12-composition-of-rotations)
function rotation_to_unitary(rotation_matrix)
    phi = acos(rotation_matrix[3,3])
    if phi == 0
        alpha = atan(-rotation_matrix[1,2], rotation_matrix[1,1])
        beta = 0
    else
        alpha = atan(rotation_matrix[1, 3], -rotation_matrix[2, 3])
        beta = atan(rotation_matrix[3, 1], rotation_matrix[3, 2])
    end
    return phase_gate(alpha)*Hadamard*phase_gate(phi)*Hadamard*phase_gate(beta)
end




#Writes a given state in its canonical form
function canonical_form(input_state; kill_Bobs_marginal = true)
    if kill_Bobs_marginal
        rho_B = partial_trace(input_state, [2, 2], 1)
        map = LinearAlgebra.sqrt(LinearAlgebra.inv(rho_B))
        input_state = kron(LinearAlgebra.I(2), map)*input_state*kron(LinearAlgebra.I(2), map)
        input_state = input_state/LinearAlgebra.tr(input_state)
    end
    T = real.([LinearAlgebra.tr(input_state*kron(Pauli_matrices[i], Pauli_matrices[j])) for i in 1:3, j in 1:3])
    if LinearAlgebra.Diagonal(T) == T
        return input_state
    else
        U, S, V = LinearAlgebra.svd(T)
        if LinearAlgebra.det(U) < 0
            U = -U
        end
        if LinearAlgebra.det(V) < 0
            V = -V
        end
        U_A = rotation_to_unitary(U')
        U_B = rotation_to_unitary(V')

        return kron(U_A, U_B)*input_state*kron(U_A', U_B')
    end
end


#Generates a random unit vector
function random_unit_vectors(n; dims = 3)
	vector = [randn(dims) for i in 1:n]
	normalized_vectors = [vector[i]/LinearAlgebra.norm(vector[i]) for i in 1:n]
	return normalized_vectors
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
function three_points_to_plane(points)
    a_vec = points[2] - points[1]
    b_vec = points[3] - points[1]
    normal_vector = LinearAlgebra.cross(a_vec, b_vec)
    offset = LinearAlgebra.dot(points[1], normal_vector)
    return normal_vector, offset
end

#Given a finite set of points, returns all triples composed of such points
function all_triples(points)
    n_points = size(points)[1]
    triples = []
    for i=1:(n_points-2)
        for j in (i+1):(n_points-1)
            for k in (j+1):n_points
                push!(triples, [points[i], points[j], points[k]])
            end
        end
    end
    return triples
end

#Given a finite set of points in 3d, all_planes gets all the planes that pass through at least three of those points
function all_planes(polytope_vertices)
    triples = all_triples(polytope_vertices)
    normal_vectors = []
    offsets = []
    for i = 1:size(triples)[1]
        new_vec, new_offset = three_points_to_plane(triples[i]) 
        push!(normal_vectors, new_vec)
        push!(offsets, new_offset)
    end
    return normal_vectors, offsets
end

#Chau's method
function critical_radius(input_state, polytope_vertices)
    normal_vectors, offsets = all_planes(polytope_vertices)
    n_vertices = size(polytope_vertices)[1]
    n_normal_vectors = length(offsets)

    canonical_state = canonical_form(input_state; kill_Bobs_marginal = true)

    a = real.([LinearAlgebra.tr(Pauli_matrices[i]*partial_trace(canonical_state, [2, 2], 2)) for i in 1:3])
    T = real.([LinearAlgebra.tr(canonical_state*kron(Pauli_matrices[i], Pauli_matrices[j])) for i in 1:3, j in 1:3])

    model = Model(Mosek.Optimizer)
    set_silent(model)

    r = @variable(model)
    @variable(model, probs[1:n_vertices] .>= 0)

    upper_bound_radius = [@expression(model, sum(probs[j]*abs(-offsets[i] + normal_vectors[i]'*polytope_vertices[j])/LinearAlgebra.norm(-offsets[i]*a + T*normal_vectors[i]) for j in 1:n_vertices)) for i in 1:n_normal_vectors]

    for i=1:n_normal_vectors
        @constraint(model, r <= upper_bound_radius[i])
    end

    @constraint(model, sum(probs) == 1)
    @constraint(model, sum(probs[j]*polytope_vertices[j] for j in 1:n_vertices) .== 0)

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


# Helper function to check if two vectors are approximately equal
approx_equal(v1, v2; atol=1e-6) = all(abs.(v1 .- v2) .< atol)

function order_polytope(polytope)
    n = length(polytope) Ã· 2
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



function simulated_annealing(objective, initial_temp, cooling_rate, max_iter,full_polytope,rho)
    current_solution = zeros(Int(length(full_polytope)/2))
    current_temp = initial_temp
    best_solution = copy(current_solution)
    current_value = objective(current_solution,full_polytope,rho)
    best_value = current_value

    for i in 1:max_iter
        new_solution = neighbor(current_solution)
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

        # Print current state
        println("Iteration $i, Temperature $current_temp, Best Value $best_value")
        if sum(current_value)<=8
            return best_solution, best_value
        end
    end
    return best_solution, best_value
end

function neighbor(x)
    y = copy(x)
    idx = rand(1:length(x))
    y[idx] = !Bool(y[idx]) % 2
    return y
end

function objective_steer(x,full_polytope,rho)
    sub_polytope = vcat([full_polytope[i] for i in 1:length(x) if x[i] == 1],[full_polytope[end+1-i] for i in 1:length(x) if x[i] == 1])
    R=critical_radius(rho,sub_polytope)
    if R==0
        return 50*length(x)
    elseif R <= 1
        return 2*sum(x)
    else
        return 2*length(x)+(R-1)
    end
end

function objective_local(x,full_polytope,rho)
    sub_polytope = vcat([full_polytope[i] for i in 1:length(x) if x[i] == 1],[full_polytope[end+1-i] for i in 1:length(x) if x[i] == 1])
    R=critical_radius(rho,sub_polytope)
    if R==0
        return 50*length(x)
    elseif R/shrinking_factor(sub_polytope)>=1
        return 2*sum(x)
    else
        return 2*length(x)+(1-R)
    end
end

function OptimizePolytope(rho,polytope,initial_temp,cooling_rate,max_iter)
    R=critical_radius(rho,polytope)

    if R/shrinking_factor(polytope)<1
        local_bool=false
        best_solution, best_value=simulated_annealing(objective_steer, initial_temp, cooling_rate, max_iter,polytope92cov,rho)

    elseif R>=1
        local_bool=true
        best_solution, best_value=simulated_annealing(objective_local, initial_temp, cooling_rate, max_iter,polytope92cov,rho)

    else
        return nothing, nothing, nothing
    end
    best_polytope = vcat([polytope[i] for i in 1:length(best_solution) if best_solution[i] == 1],[polytope[end+1-i] for i in 1:length(best_solution) if best_solution[i] == 1])
    return best_solution, best_polytope, local_bool
end


