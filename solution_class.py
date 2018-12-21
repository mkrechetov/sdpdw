import numpy as np

class Solution():
    def __init__(self, lower_bound, upper_bound, assignment, known, V):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.assignment = assignment
        self.known = known
        self.V = V
        
    def __str__(self):
        return "lower_bound =\n %s,\n upper_bound =\n %s,\n assignment = %s, variables = %s,"% (self.lower_bound, self.upper_bound, self.assignment, self.known)
    
    def __lt__(self, other):
        return self.lower_bound > other.lower_bound
    
    def Unknown(self):
        ind_unknown = np.argwhere(np.isnan(self.known))
        ind_unknown = np.reshape(ind_unknown, (1, len(ind_unknown)))[0]
        ind_unknown = ind_unknown.tolist() #list of indices to be determined
        return ind_unknown