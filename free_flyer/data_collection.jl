using Distributed
addprocs(2)

@everywhere include("problem.jl")
@everywhere include("../fileIO.jl")
@everywhere using LinearAlgebra, Random, ProgressMeter

@everywhere function findIC(obstacles,posmin,posmax,velmin,velmax)
    IC_found = false
    x0 = deepcopy([posmin;velmin])
    while !IC_found
        x0 = [posmin .+ (posmax-posmin).*rand(2); velmin .+ (velmax-velmin).*rand(2)]
        if !any([x0[1] >= obstacle[1][1] && x0[1] <= obstacle[1][2] && 
                    x0[2] >= obstacle[2][1] && x0[2] <= obstacle[2][2] for obstacle in obstacles])
            IC_found = true
        end
    end
    return x0
end

@everywhere function findObs(x0,Nobstacles,posmin,posmax,min_box_size,max_box_size,border_size,box_buffer; max_iter=50)
    obs = []
    itr = 0
    while length(obs) < Nobstacles && itr < max_iter
        xmin = (posmax[1] - border_size - max_box_size)*rand() + border_size
        xmin = max(xmin, x0[1])
        xmax = xmin + min_box_size  + (max_box_size - min_box_size)*rand()
        ymin = (posmax[2] - border_size - max_box_size)*rand() + border_size
        ymin = max(ymin, x0[2])
        ymax = ymin + min_box_size  + (max_box_size - min_box_size)*rand()
        obstacle = [[xmin - box_buffer, xmax + box_buffer], 
                    [ymin - box_buffer, ymax + box_buffer]]
        if is_free_state(x0, [obstacle])
            push!(obs, obstacle)
        end
        itr+=1
    end
    if length(obs) != Nobstacles
        return []
    end
    obs
end

@everywhere function is_free_state(x0,obstacles)
    if any([x0[1] >= obstacle[1][1] && x0[1] <= obstacle[1][2] && 
                    x0[2] >= obstacle[2][1] && x0[2] <= obstacle[2][2] for obstacle in obstacles])
        return false
    end
    return true
end

@everywhere function make_dataset(dir_idx;Ndata=5000,Nobstacles=8)
    Random.seed!(dir_idx)
    # Parameters
    n = 2;
    Ac = [zeros(n,n) I;
          zeros(n,2*n)]
    Bc = [zeros(n,n) I]'

    dh = 0.75
    A, B = c2d(Ac, Bc, dh)

    N =6 # horizon
    m = size(B, 2)
    Q = diagm([2.;2;1;1])
    R = 0.1*Matrix{Float64}(I,n,n)
    mass_ff_min = 15.36
    mass_ff_max = 18.08
    mass_ff = 0.5*(mass_ff_min+mass_ff_max)
    thrust_max = 2*1.  # max thrust [N] from two thrusters 
    umin = -thrust_max/mass_ff
    umax = thrust_max/mass_ff
    velmin = -0.2
    velmax = 0.2
    posmin = [0;0]
    ft2m = 0.3048
    posmax = ft2m*[12.,9.]
    max_box_size = 0.75
    min_box_size = 0.25
    box_buffer = 0.025
    border_size = 0.05
    UB, solved = upperBoundObstacleAvoidanceProb()
    
    obs_new_ct = 10
    toggle_ct = 50

    x0 = [ft2m*ones(2); zeros(2)]
    xg = [0.9*posmax; zeros(n)]
    obstacles = []
    
    # Problem
    probdata = Dict("A"=>A, "B"=>B, "N"=>N, 
            "umin"=>umin, "umax"=>umax, 
            "velmin"=>velmin, "velmax"=>velmax, 
            "posmin"=>posmin, "posmax"=>posmax, 
            "Q"=>Q, "R"=>R,
            "max_box_size"=>max_box_size, "min_box_size"=>min_box_size,
            "border_size"=>border_size, "UB"=>UB, "box_buffer"=>box_buffer);

    X0 = []
    Xg = []
    O = []
    X = []
    Y = []
    U = []
    J = Float64[]
    solution_times = Float64[]
    node_counts = Int[]

    ii = 0
    ii_obs = 0

    toggle_obstacles = true
    ii_toggle = 0 

    while ii < Ndata
        if toggle_obstacles
            if rand() < 0.5
                x0 = [posmin .+ (posmax-posmin).*rand(2); velmin .+ (velmax-velmin).*rand(2)]
            else
                x0 = [posmin .+ 0.5*(posmax-posmin).*rand(2); velmin .+ (velmax-velmin).*rand(2)]
            end

            obstacles = findObs(x0, Nobstacles,posmin,posmax,min_box_size,max_box_size,border_size,box_buffer)
            if length(obstacles) == 0
                continue
            end

            if mod(ii_toggle,obs_new_ct) == 0
                toggle_obstacles = false
                ii_obs = 0
                ii_toggle = 0
            end
        else
            # Generate obstacles
            if mod(ii_obs,obs_new_ct) == 0
                obstacles = []
                for _ in 1:Nobstacles
                    xmin = (posmax[1] - border_size - max_box_size)*rand() + border_size
                    xmax = xmin + min_box_size  + (max_box_size - min_box_size)*rand()
                    ymin = (posmax[2] - border_size - max_box_size)*rand() + border_size
                    ymax = ymin + min_box_size  + (max_box_size - min_box_size)*rand()
                    obstacle = [[xmin - box_buffer, xmax + box_buffer], 
                                [ymin - box_buffer, ymax + box_buffer]]
                    push!(obstacles, obstacle)
                end
            end

            # Generate initial condition (that is outside of obstacles)
            x0 = findIC(obstacles,posmin,posmax,velmin,velmax)

            if mod(ii_toggle,toggle_ct) == 0
                toggle_obstacles = true
                ii_obs = 0
                ii_toggle = 0
            end
        end

        # Solve Problem
        x, u, y, cost, solve_time, node_count, solver_success = solveObstacleAvoidanceProb(A, B, Q, R, N, 
                                                        umin, umax, velmin, velmax, 
                                                        posmin, posmax, obstacles, x0, xg);

        # Save all the data
        if solver_success
            push!(X0, x0)
            push!(Xg, xg)
            push!(O, obstacles)
            push!(X, x)
            push!(Y, y)
            push!(U, u)
            push!(J, cost)
            push!(solution_times, solve_time)
            push!(node_counts, node_count)

            ii += 1
            ii_obs += 1
            ii_toggle += 1
        end
    end

    base = "data/testdata"
    filename = string(base, string(dir_idx), ".jld")

    training_data = Dict("X0"=>X0, "Xg"=>Xg, "O"=>O, "X"=>X, "Y"=>Y,
        "U"=>U, "J"=>J, 
        "solve_time"=>solution_times, "node_count"=>node_counts,
        "prob"=>probdata)
    writeTrainingData(filename, training_data)
    return
end

Nsets = 2
dir_idxs = [18+ii for ii in 1:Nsets]
time_parallel = @elapsed results = @showprogress pmap(make_dataset, dir_idxs)    
