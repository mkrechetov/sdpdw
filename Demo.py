
# coding: utf-8

# In[1]:

import numpy as np

from bqp_class import *
from sdp_class import *
from bab_class import *


# In[4]:

#uploading chimera instance from file

problem = bqp("From File", filename="ran2f-b.json")
sdp = SDP(problem, maxrounds = 100)
print(sdp)


# In[8]:

#generating random instance

problem = bqp("Random", n=10, p=0.4, seed = 1)
print(problem)
sdp = SDP(problem, rank = 2)
print(sdp)


# In[ ]:



