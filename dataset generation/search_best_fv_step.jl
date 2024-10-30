<<<<<<< HEAD
include("functions.jl")
using DataFrames, Arrow, Dates
using Base.Threads


# Load the Arrow files
rearranged_vectors_table_92 = Arrow.Table("polytope92cov_rearranged.arrow")

# Convert to regular Julia arrays
rearranged_vectors_92 = collect(rearranged_vectors_table_92.rearranged_vectors)

# Optionally, convert each element to a regular Array for better usability
polytope92cov = [convert(Vector{Float64}, collect(x)) for x in rearranged_vectors_92]



batch_size = 50
num_batches = 1

# annealing settings
initial_temp = 2
cooling_rate = 0.985 
max_iter = 500
# cooling_rate = 0.997
# max_iter = 1300


function create_row(polytope,step)
    rho = rho_Bures(4)
    # This variable tells if the state is separable
    separable_bool = is_separable(rho)
    if separable_bool
        local_bool = true
        optimal_polytope = nothing
        inner_radius = nothing
        outer_radius = nothing
    else
        optimal_polytope, local_bool,inner_radius,outer_radius = OptimizePolytopeFreeVec(rho, polytope, initial_temp, cooling_rate, max_iter,step)
    end
    return (State = rho, 
            Separable = separable_bool, 
            Local = local_bool, 
            Polytope = optimal_polytope,
            InnerRadius = inner_radius, 
            OuterRadius = outer_radius)
end

# Loop through steps from 0.5 to 0.05, decrementing by 0.05
for step in 0.5:-0.05:0.05
    println("Statring generating values for Step=$step")
    for row in 1:batch_size
        create_row(polytope92cov, step)
    end
=======
include("functions.jl")
using DataFrames, Arrow, Dates
using Base.Threads


# Load the Arrow files
rearranged_vectors_table_92 = Arrow.Table("polytope92cov_rearranged.arrow")

# Convert to regular Julia arrays
rearranged_vectors_92 = collect(rearranged_vectors_table_92.rearranged_vectors)

# Optionally, convert each element to a regular Array for better usability
polytope92cov = [convert(Vector{Float64}, collect(x)) for x in rearranged_vectors_92]



batch_size = 50
num_batches = 1

# annealing settings
initial_temp = 2
cooling_rate = 0.985 
max_iter = 500
# cooling_rate = 0.997
# max_iter = 1300


function create_row(polytope,step)
    rho = rho_Bures(4)
    # This variable tells if the state is separable
    separable_bool = is_separable(rho)
    if separable_bool
        local_bool = true
        optimal_polytope = nothing
        inner_radius = nothing
        outer_radius = nothing
    else
        optimal_polytope, local_bool,inner_radius,outer_radius = OptimizePolytopeFreeVec(rho, polytope, initial_temp, cooling_rate, max_iter,step)
    end
    return (State = rho, 
            Separable = separable_bool, 
            Local = local_bool, 
            Polytope = optimal_polytope,
            InnerRadius = inner_radius, 
            OuterRadius = outer_radius)
end

# Loop through steps from 0.5 to 0.05, decrementing by 0.05
for step in 0.5:-0.05:0.05
    println("Statring generating values for Step=$step")
    for row in 1:batch_size
        create_row(polytope92cov, step)
    end
>>>>>>> bc76b166fdd330aab41ccc0fee09d33d38c643ea
end