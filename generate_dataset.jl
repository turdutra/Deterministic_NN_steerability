include("functions.jl")
using DataFrames, Arrow, Dates
using Base.Threads

file_polytope92cov = parse.(Float64, readlines("polytope92cov.txt")[1:end])
polytope92cov = [[file_polytope92cov[i], file_polytope92cov[i+1], file_polytope92cov[i+2]] for i in 1:3:length(file_polytope92cov)]
polytope92cov = order_polytope(polytope92cov)

# Initialize the CriticalRadius column with NaN values

batch_size = 500
num_batches = 20

# annealing settings
initial_temp = 1
cooling_rate = 0.997
max_iter = 1000
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

function process_row(polytope, result_channel)

    try
        new_row = create_row(polytope)
        put!(result_channel, (
            State = new_row.State, 
            Separable = new_row.Separable, 
            Local = new_row.Local, 
            PolytopeBin = new_row.PolytopeBin, 
            Polytope = new_row.Polytope,
            InnerRadius = new_row.InnerRadius, 
            OuterRadius = new_row.OuterRadius
        ))
    catch e
        println("Error", e)
        # Print the detailed stack trace
        Base.showerror(stderr, e)
        println(stderr, catch_backtrace())
    end
   
    end
    
function generate_batch(batch, polytope,result_channel)

    tasks = []
    start_time = Dates.now()  # Start time for the batch

    for i in 1:batch_size
        task = Threads.@spawn process_row(polytope, result_channel)
        push!(tasks, task)
    end

    # Wait for all tasks to complete
    for task in tasks
        fetch(task)
    end
    end_time = Dates.now()  # End time for the batch
    elapsed_time = end_time - start_time
    println("Batch $batch took $(Dates.value(elapsed_time) / 1000) seconds to complete.\n")


    # Initialize an empty DataFrame with the appropriate column types
    df = DataFrame(
        State=Vector{Matrix{ComplexF64}}(),                    # State is a vector of complex matrices
        Separable=Vector{Bool}(),                              # Separable is a vector of Bool
        Local=Vector{Union{Bool, Missing}}(),                  # Local is a vector that can be Bool or Missing
        PolytopeBin=Vector{Union{Vector{Bool}, Nothing, Missing}}(),   # PolytopeBin is a vector that can be a binary vector, Nothing, or Missing
        Polytope=Vector{Union{Vector{Vector{Float64}}, Nothing, Missing}}(),  # Polytope is a vector that can be a vector of vectors, Nothing, or Missing
        InnerRadius=Vector{Union{Float64, Nothing, Missing}}(),  # InnerRadius is a vector that can be a Float, Nothing, or Missing
        OuterRadius=Vector{Union{Float64, Nothing, Missing}}()   # OuterRadius is a vector that can be a Float, Nothing, or Missing
    )
    # Consume the channel and populate the DataFrame
    for _ in 1:batch_size
        new_row = take!(result_channel)
        push!(df, new_row)
    end

    # Save the DataFrame
    println("Saving DataFrame for batch $batch")
    Arrow.write("dataset_b$batch", df)

    # Clear the channel
    close(result_channel)

    # Call garbage collector
    GC.gc()
end

for batch in 1:num_batches
    result_channel = Channel{NamedTuple}(batch_size)
    println("Starting to process $batch")

    # Process the batch
    generate_batch(batch, polytope92cov,result_channel)
end
