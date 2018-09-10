#!/usr/bin/env julia

#julia 0.6.2

using ArgParse
using JSON

using Distributions
using CPUTime

function main(parsed_args)
    #read the problem
    file = open(parsed_args["input-file"])
    time_limit = parsed_args["time-limit"]

    data = JSON.parse(file)
    version = data["version"]
    id = data["id"]
    metadata = data["metadata"]
    variable_ids = data["variable_ids"]
    variable_domain = data["variable_domain"]
    scale = data["scale"]
    offset = data["offset"]
    linear_terms = data["linear_terms"]
    quadratic_terms = data["quadratic_terms"]
    if haskey(data, "description")
        description = data["description"]
    end
    if haskey(data, "solutions")
        solutions = data["solutions"]
    end

    #extract objective function
    n = size(variable_ids, 1);
    #transform variable ids to 1:n
    variables = Dict(variable_ids[i] => i for i = 1:n)

    #form matrix A from quadratic terms
    A = zeros(n, n);

    for quad_iter = 1:size(quadratic_terms)[1]
        i = variables[quadratic_terms[quad_iter]["id_head"]];
        j = variables[quadratic_terms[quad_iter]["id_tail"]];
        cij = quadratic_terms[quad_iter]["coeff"];
        A[i, j] = cij;
        A[j, i] = cij;
    end  

    #form column-vector b from linear terms
    b = zeros(n, 1);
    for lin_iter = 1:size(linear_terms)[1]
        i = variables[linear_terms[lin_iter]["id"]];
        h = linear_terms[lin_iter]["coeff"];
        b[i, 1] = h;
    end 

    #transform it to spin problem without linear term

    if (variable_domain == "spin") 
        if b == zeros(n, 1) 
            Cost_Matrix = A/2; 
            dimension = n; 
        else
            Cost_Matrix = [A/2 b/2; transpose(b)/2 0]; 
            dimension = n+1; 
        end
    elseif (variable_domain == "boolean")
        B = A/2+Diagonal(vec(b)); 
        c = B*ones(n, 1); 
        d = ones(1, n)*B*ones(n, 1);
        Cost_Matrix = [B c; transpose(c) d]/4; 
        dimension = n+1; 
    end


    #--------------------------------Burer-Monteiro-----------------------
    rank_constraint = trunc.(Int, sqrt(2*n));
    #parameters
    time_limit = 20;

    #Intialize randomly on sphere_k
    V = rand(Normal(), rank_constraint, dimension);
    #normalize columns
    v = [1/norm(V[:, col], 2) for col in 1:dimension];
    V = V.*v';

    Lconst = norm(vec(Cost_Matrix), 2);
    step = 1/(Lconst);

    t1 = time_ns();
    while true
        for iter = 1:1000
            gradient = 2*V*Cost_Matrix; 

            V = V - step*gradient;

            v = [1/norm(V[:, col], 2) for col in 1:dimension];
            V = V.*v';
        end
    
    
        #update stopping variables
        lambda = zeros(1, dimension);
        M = V*Cost_Matrix;
        lambda = M[1, :]./V[1, :];

        if (norm(M - V.*lambda', Inf) < 0.5) & (eigvals(Cost_Matrix - diagm(vec(lambda)))[1] > -0.1)
            break;
        end    

        if ((time_ns()-t1)/1.0e9 > time_limit)
            println("timeover");
            break;
        end
    end    

    t2 = time_ns();
    elapsedTime = (t2 - t1)/1.0e9;

    time_remain = time_limit - (elapsedTime);
    #--------------------------------------------------------
    
    #randomized-rounding

    rounding_trials = 1000;
    cutAssignment = zeros(1, dimension);
    cutVal = -1;

    t=time_ns()
    for cut_trials = 1:rounding_trials
        r = rand(Normal(), 1, rank_constraint);
        r = r/sqrt(dot(r, r));
        cut = zeros(Float64, dimension, 1);
        for cut_iter = 1:dimension
           cut[cut_iter, 1] =  sign(dot(r, V[:, cut_iter]));
        end

        cutValnew = transpose(cut)*Cost_Matrix*cut; 

        if (cut_trials == 1)
            cutVal = cutValnew;
            cutAssignment = cut;
        elseif (cutValnew[1, 1] < cutVal[1, 1])
            cutVal = cutValnew;
            cutAssignment = cut;
        end

        # rounding for remaining time
        #if (cut_trials > 1000)
        #    if (time()-t > time_remain)
        #       break;
        #    end
        #end  
    end

    for i = 1:dimension
        if (cutAssignment[i] == 0)
            cutAssignment[i] = sign.(rand(Normal(), 1, 1)[1]);    
        end    
    end

    if (dimension != n)
        #make extra variable equal 1
        cutAssignment = cutAssignment*cutAssignment[dimension];
    end
    cutAssignment = trunc.(Int, cutAssignment);


    total_time = (time_ns() - t)/1.0e9 + elapsedTime;


    for i = 1:dimension
        if (cutAssignment[i] == 0)
            cutAssignment[i] = sign.(rand(Normal(), 1, 1)[1])
        end
    end
    
    if (dimension != n)
        #make extra variable equal 1
        cutAssignment = cutAssignment*cutAssignment[dimension];
    end
    cutAssignment = trunc.(Int, cutAssignment);
    #evaluation = scale*(transpose(cutAssignment)*Cost_Matrix*cutAssignment + offset);

    #form bqpjson solution to the bqp file
    result = vec(cutAssignment);

    if (dimension == n + 1)
        deleteat!(result, dimension);
    end
    if (variable_domain == "boolean")
        result = (result+ones(n, 1))/2;
        result = trunc.(Int, result);
    end
    
    evaluation = (transpose(result)*A/2*result + transpose(b)*result);

    # print the results
    if parsed_args["show-solution"]
        println(result)
    end

    nodes = length(variable_ids)
    edges = length(quadratic_terms)

    lt_lb = -sum(abs(lt["coeff"]) for lt in linear_terms)/scale
    qt_lb = -sum(abs(qt["coeff"]) for qt in quadratic_terms)/scale
    lower_bound = sum(diag(Cost_Matrix*(transpose(V)*V)))

    cut_count = 0

    best_objective = evaluation[1]
    best_nodes = 0
    best_runtime = total_time
    scaled_objective = scale*(best_objective+offset)
    scaled_lower_bound = scale*(lower_bound+offset)

    println("BQP_DATA, $(nodes), $(edges), $(scaled_objective), $(scaled_lower_bound), $(best_objective), $(lower_bound), $(best_runtime), $(cut_count), $(best_nodes)")

end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input-file", "-f"
            help = "the data file to operate on (.json)"
            required = true
        "--time-limit", "-t"
            help = "puts a time limit on the sovler"
            arg_type = Float64
            default = 10.0
        "--show-solution", "-s"
            help = "print the solution"
            action = :store_true
    end

    return parse_args(s)
end

if isinteractive() == false
    main(parse_commandline())
end

