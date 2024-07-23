function objective(x)
    polytope = vcat([polytope72cov_ordered[i] for i in 1:length(x) if x[i] == 1],[polytope72cov_ordered[73-i] for i in 1:length(x) if x[i] == 1])
    #s_factor=shrinking_factor(polytope)
    R=critical_radius(steering_df[2, :State],polytope)
    return ifelse(R >= 1, 2*sum(x), 2*length(x)+(1-R))
end
function neighbor(x)
    y = copy(x)
    idx = rand(1:length(x))
    y[idx] = !Bool(y[idx]) % 2
    return y
end

function simulated_annealing(objective, initial_solution, initial_temp, cooling_rate, max_iter)
    current_solution = initial_solution
    current_temp = initial_temp
    best_solution = copy(initial_solution)
    current_value = objective(initial_solution)
    best_value = current_value

    for i in 1:max_iter
        new_solution = neighbor(current_solution)
        new_value = objective(new_solution)
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
        if sum(current_value)<=6
            return best_solution, best_value
        end
    end

    return best_solution, best_value
end


initial_solution = ones(36)

initial_temp = 2000
cooling_rate = 0.98
max_iter = 1000

# Run simulated annealing
best_solution, best_value = simulated_annealing(objective, initial_solution, initial_temp, cooling_rate, max_iter)

println("Best Solution: $best_solution")
println("Best Value: $best_value")