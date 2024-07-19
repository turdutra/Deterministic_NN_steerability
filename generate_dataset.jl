

using DataFrames, Arrow, Dates



# Initialize the CriticalRadius column with NaN values

batch_size = 3000
num_batches=3

# Define the initial DataFrame structure
df = DataFrame(State = Matrix{ComplexF64}[], 
               Separable = Bool[], 
               Local = Bool[], 
               PolytopeBin = Union{Vector{Int}, Nothing}[], 
               Polytope = Union{Vector{Vector{Float64}}, Nothing}[])

function create_row(polytope)
    rho=rho_Bures(4)
    #This variable tells if the state is separable
    separable_bool=is_separable(rho)
    if separable_bool
        local_bool=true
        optimal_polytope_bin=nothing
        optimal_polytope=nothing
    else
        optimal_polytope_bin,optimal_polytope,local_bool=OptimizePolytope(rho,polytope)
    end
    return (State = rho, 
    Separable = separable_bool, 
    Local = local_bool, 
    PolytopeBin = optimal_polytope_bin, 
    Polytope = optimal_polytope)
end

function generate_batch(batch)    
    tasks = []
    start_time = Dates.now()  # Start time for the batch

    for i in 1:batch_size
        task = Threads.@spawn begin
            try
                new_row=create_row(polytope)
                push!(results_df, new_row)
            catch e
                println("Error processing row $i, batch $batch: ", e)
            end
        end
        push!(tasks, task)
    end

    # Wait for all tasks to complete
    for task in tasks
        fetch(task)
    end

    end_time = Dates.now()  # End time for the batch
    elapsed_time = end_time - start_time
    print("Batch $batch took (Dates.value($elapsed_time)/1000) seconds to complete.\n")

    # Save the DataFrame
    Arrow.write("dataset_b$batch", df)

    # Call garbage collector
    GC.gc()
end

for batch in 1:num_batches
    print("Starting to process $batch")

    # Process the batch
    generate_batch( batch)
end
