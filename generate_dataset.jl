include("functions.jl")
using DataFrames, Arrow, Dates
using Base.Threads

file_polytope92cov = parse.(Float64, readlines("polytope92cov.txt")[1:end])
polytope92cov = [[file_polytope92cov[i], file_polytope92cov[i+1], file_polytope92cov[i+2]] for i in 1:3:length(file_polytope92cov)]
polytope92cov = order_polytope(polytope92cov)

# Initialize the CriticalRadius column with NaN values

batch_size = 1
num_batches = 1

# annealing settings
initial_temp = 2500
cooling_rate = 0.97
max_iter = 300

#initial_temp = 2000
#cooling_rate = 0.98
#max_iter = 1000


function create_row(polytope)
    rho = rho_Bures(4)
    # This variable tells if the state is separable
    separable_bool = is_separable(rho)
    if separable_bool
        local_bool = true
        optimal_polytope_bin = nothing
        optimal_polytope = nothing
        inner_radius = nothing
        outer_radius = nothing
    else
        optimal_polytope_bin, optimal_polytope, local_bool,inner_radius,outer_radius = OptimizePolytope(rho, polytope, initial_temp, cooling_rate, max_iter)
    end
    return (State = rho, 
            Separable = separable_bool, 
            Local = local_bool, 
            PolytopeBin = optimal_polytope_bin, 
            Polytope = optimal_polytope,
            InnerRadius = inner_radius, 
            OuterRadius = outer_radius)
end

function process_row(i, batch, polytope, df)

    new_row = create_row(polytope)
    df.State[i] = new_row.State
    df.Separable[i] = new_row.Separable
    df.Local[i] = new_row.Local
    df.PolytopeBin[i] = new_row.PolytopeBin
    df.Polytope[i] = new_row.Polytope
    df.InnerRadius[i] = new_row.InnerRadius
    df.OuterRadius[i] = new_row.OuterRadius

end

function generate_batch(batch, polytope)
    # Define the DataFrame structure with preallocated rows
    df = DataFrame(State = Union{Matrix{ComplexF64}, Missing}[missing for _ in 1:batch_size], 
                   Separable = Union{Bool, Missing}[missing for _ in 1:batch_size], 
                   Local = Union{Bool, Missing}[missing for _ in 1:batch_size], 
                   PolytopeBin = Union{Vector{Int}, Nothing, Missing}[missing for _ in 1:batch_size], 
                   Polytope = Union{Vector{Vector{Float64}}, Nothing, Missing}[missing for _ in 1:batch_size],
                   InnerRadius = Union{Float64, Nothing, Missing}[missing for _ in 1:batch_size], 
                   OuterRadius = Union{Float64, Nothing, Missing}[missing for _ in 1:batch_size])
                   
                   
    
    tasks = []
    start_time = Dates.now()  # Start time for the batch

    for i in 1:batch_size
        let i = i  # Capture `i` for the closure
            task = Threads.@spawn process_row(i, batch, polytope, df)
            push!(tasks, task)
        end
    end

    # Wait for all tasks to complete
    for task in tasks
        fetch(task)
    end
    end_time = Dates.now()  # End time for the batch
    elapsed_time = end_time - start_time
    println("Batch $batch took $(Dates.value(elapsed_time) / 1000) seconds to complete.\n")

    # Save the DataFrame
    Arrow.write("dataset_b$batch", df)

    # Call garbage collector
    GC.gc()
end

for batch in 1:num_batches
    println("Starting to process $batch")

    # Process the batch
    generate_batch(batch, polytope92cov)
end
