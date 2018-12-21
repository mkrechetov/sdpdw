import numpy as np
from scipy.sparse import csr_matrix 
import networkx as nx
import json
import copy

class bqp: #to read and to rewrite problems, return matrix
      
    def __init__(self, init_type, **kwargs): #init_type: {Directly, Random, From File, Random Tree, Random Chimera}
        
        
        if (init_type == "Directly"):
            if 'Const' in kwargs.keys():
                Const = kwargs['Const']
            else:
                Const = 0.0
            
            if 'Pot' in kwargs.keys():
                Pot = kwargs['Pot']
            else:
                Pot = np.zeros((1, len(kwargs['Inter'])))
                
            (self.Inter, self.Pot, self.Const) = (kwargs['Inter'], Pot, Const)
        elif (init_type == "Random"):
            
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                print("Should specify number of vertices n=")
                
            if 'p' in kwargs.keys():
                p = kwargs['p']
            else:
                print("Should specify the probability of an edge p=")    
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None

            G = nx.gnp_random_graph(n, p, seed)
            A = csr_matrix.todense(nx.adjacency_matrix(G))

            (self.Inter, self.Pot, self.Const) = (Laplacian(A)/4, np.zeros((n)), 0)
            
        elif (init_type == "From File"):
            
            if 'filename' in kwargs.keys():
                filename = kwargs['filename']
            else:
                print("You should specify the filename!")
                
            import os
            name, extension = os.path.splitext(filename)

            if (extension == '.json'):
                (self.Inter, self.Pot, self.Const) = BQPJSON(filename)
                self.Inter = 1*self.Inter
                self.Pot = 1*self.Pot
                self.Const = 1*self.Const
            #elif (file_extension == '.mat'):
                #retrieve a dense graph from .mat file
            #elif (file_extension == '.sparse'):
                #retrieve a sparse graph from .sparse file
            else:
                print("Wrong File Extension")
                
        elif (init_type == "Random Chimera"):
            import dwave_networkx as dnx

            G = dnx.chimera_graph(kwargs['M'], kwargs['N'], kwargs['L'])
            A = csr_matrix.todense(nx.adjacency_matrix(G))

            (self.Inter, self.Pot, self.Const) = (Laplacian(A)/4, np.zeros((n)), 0)
        
        elif (init_type == "Random Tree"):
            
            if 'seed' in kwargs.keys():
                seed = kwargs['seed']
            else:
                seed = None
    
            if 'n' in kwargs.keys():
                n = kwargs['n']
            else:
                n = random.randint(10, 100)

            G = nx.random_tree(n, seed)
            A = csr_matrix.todense(nx.adjacency_matrix(G))
            
            (self.Inter, self.Pot, self.Const) = (Laplacian(A)/4, np.zeros((n)), 0)
            
        
    def __str__(self):
        return "Matrix of Interactions =\n %s,\n Vector of Potentials =\n %s,\n Constant = %s"% (self.Inter, self.Pot, self.Const)    
    
    def Evaluate(self, assignment):
        return np.trace(np.matmul(self.Inter, np.dot(assignment.T, assignment))) + np.dot(self.Pot, assignment.T) + self.Const
            
        
    #converts to quadratic problem without linear and constant terms
    #s.t. we will have  maximization  <x, C x> subject to spin constraints    
    def Matrix(self):
        
        n = len(self.Inter)
        
        C = np.zeros((n + 1, n + 1))
        C[np.ix_(list(range(n)),list(range(n)))] = self.Inter
        half_Pot = [i/2 for i in self.Pot]
        C[np.ix_([n],list(range(n)))] = half_Pot
        C[np.ix_(list(range(n))), [n]] = half_Pot

        C[np.ix_([n], [n])] = self.Const
            
        return C
    
    def Merge(self, partial_assignment, curr_known):
        
        variables = copy.deepcopy(curr_known)
        variables.reshape(1, len(variables))
        
        ind_unknown = np.argwhere(np.isnan(variables))
        ind_unknown = np.reshape(ind_unknown, (1, len(ind_unknown)))[0]
        ind_unknown = ind_unknown.tolist() 
        
        variables[np.ix_(ind_unknown)] = partial_assignment.reshape(1, (len(partial_assignment[0])))
        return variables.reshape((1, len(curr_known)))
    
    def InducedProblem(self, variables): #variables is (1, n) array of +-1s and nans.
        
        ind_known = np.argwhere(~np.isnan(variables))
        ind_known = np.reshape(ind_known, (1, len(ind_known)))[0]
        ind_known = ind_known.tolist()    

        ind_unknown = np.argwhere(np.isnan(variables))
        ind_unknown = np.reshape(ind_unknown, (1, len(ind_unknown)))[0]
        ind_unknown = ind_unknown.tolist() 

        
        #induced Interaction   
        induced_Inter = self.Inter[np.ix_(ind_unknown,ind_unknown)]    
        known_Inter = self.Inter[np.ix_(ind_known,ind_known)]
        cross_Inter = self.Inter[np.ix_(ind_known, ind_unknown)]
        
        var_defined = variables[np.argwhere(~np.isnan(variables))]
        var_defined = np.reshape(var_defined, (1, len(var_defined)))
        #print(var_defined)
        
        known_Potentials = self.Pot[np.argwhere(~np.isnan(variables))]
        known_Potentials = np.reshape(known_Potentials, (1, len(known_Potentials)))
        
        
        unknown_Potentials = self.Pot[np.argwhere(np.isnan(variables))]
        unknown_Potentials = np.reshape(unknown_Potentials, (1, len(unknown_Potentials)))
        
        #print("lin1:", 2*np.matmul(var_defined,cross_Inter))
        #print("lin2:", unknown_Potentials)
        linear_term = 2*np.matmul(var_defined,cross_Inter) + unknown_Potentials
        #linear_term = np.asarray(linear_term[0].tolist())
        #linear_term = linear_term.reshape((1, len(ind_unknown)))
        #print("pampam:", np.dot(var_defined.T, var_defined))
        
        lyft = np.trace(np.matmul(known_Inter, np.dot(var_defined.T, var_defined))) + np.dot(var_defined, known_Potentials.T) + self.Const
        
        return bqp(init_type = "Directly", Inter = induced_Inter, Pot = linear_term, Const = lyft)
        
            
#------------------------------------------------------------------------------------------------------------------
     
def Laplacian(Adjacency): 
    
    G = nx.from_numpy_matrix(Adjacency)
    L = csr_matrix.todense(nx.laplacian_matrix(G))

    return L


def Adjacency(Laplacian): 
    
    A = copy.deepcopy(Laplacian)
    np.fill_diagonal(A, 0)
    A = -1*A
    
    return A
    
        
#implement bqpjson parsing

def BQPJSON(filename):
    #read from bqpjson 
   
    file = open(filename).read();#open("ran1_b_1.json", "r")
    data = json.loads(file)
    version = data["version"]
    ids = data["id"]
    metadata = data["metadata"]
    variable_ids = data["variable_ids"]
    variable_domain = data["variable_domain"]
    scale = data["scale"]
    offset = data["offset"]
    linear_terms = data["linear_terms"]
    quadratic_terms = data["quadratic_terms"]
    #if haskey(data, "description")
    #description = data["description"]
    #end 
    #if haskey(data, "solutions")
    #solutions = data["solutions"]
    #end

    n = len(variable_ids)
    #transform variable ids to 1:n
    variables = {}
    for i in range(n):
        variables.update({variable_ids[i] : i}) 


    #form matrix A from quadratic terms
    A = np.zeros((n,n), dtype=float)
    for quad_iter in range(len(quadratic_terms)):
        i = variables[quadratic_terms[quad_iter]["id_head"]]
        j = variables[quadratic_terms[quad_iter]["id_tail"]]
        cij = quadratic_terms[quad_iter]["coeff"]
        A[i, j] = -cij/2
        A[j, i] = -cij/2
        
    #form column-vector b from linear terms
    b = np.zeros((n), dtype=float)
    for lin_iter in range (len(linear_terms)):
        i = variables[linear_terms[lin_iter]["id"]]
        h = linear_terms[lin_iter]["coeff"]
        b[i] = -h
    
    # all in all we have maximization    <x, A x> + <b, x> subjecti to boolean/spin constraints
    
    #------------------------------converting to spin problem-----------------------------------
    if (variable_domain == "boolean"):  #Check it!!!
        Inter = A/4
        Pot = b/2 + np.matmul(np.ones(n), A/2)
        c = np.matmul(np.ones(n), b/2)+np.matmul(np.ones(n), np.matmul(A/4, np.ones(n)))
    else:
        Inter = A
        Pot = b
        c = 0.0
    #-------------------------------------------------------------------------------------------    
    
    return (Inter, Pot, c)
        
        
#"""
#-------------------------------------------------------------------
#maximize   <x, Inter x> + <Pot, x> + Const
#subject to spin constraints
#-------------------------------------------------------------------
#"""
