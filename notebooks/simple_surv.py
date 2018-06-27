'''

Trivial survial problem

Create a survival time:
1) Without any relation to X
2) with linear relation to X
3) with non-linear relation to X

Test out CoxPH model on each, confirm results
x ~ Poisson(lambda=5)

Generate 3,000 data points of x sampled from this distribution

** adjust time to get 40% of actual events ?? **

Create y value 

non-linear example:

T ~ Weibull(lambda=0.1*nu(X),p=2.1)
where nu(X) = (-0.5*x_1 + 9*x_2 + 19*x_3)^2

'''
import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(20180531) #for reproducability

import pandas as pd
#lifelines CoxPH model
from lifelines import CoxPHFitter

import ipdb

def gen_survival(x,x_relation=None):
    ## np returns 1-param weibull, see: doc for 2 param
    if x_relation == 'lin':
        T = x * np.random.weibull(a=2.1,size=3000)
    elif x_relation == 'sq':
        ## alternatly, expcted mean from the distribution of X
        T =  ( (x-x.mean())**2 ) * np.random.weibull(a=2.1,size=3000)
    else:
        T = np.random.weibull(a=2.1,size=3000)

    ## random censoring
    event_observed = np.zeros_like(T)
    event_observed[np.random.rand(3000)<0.4] = 1
    
    toy_dataset = pd.DataFrame([x,T,event_observed]).transpose()
    toy_dataset.columns = ['x','T','event_obs']
    
    return toy_dataset


## have a look at the relation of X and the log(T)
x = np.random.poisson(lam=10,size=3000)
toy_none = gen_survival(x,x_relation=None)
toy_lin  = gen_survival(x,x_relation='lin')
toy_sq   = gen_survival(x,x_relation='sq')

plt.scatter(toy_none.x.values,np.log(toy_none['T'].values),alpha=0.25)
plt.scatter(toy_lin.x.values, np.log(toy_lin['T'].values),alpha=0.25)
plt.scatter(toy_sq.x.values,  np.log(toy_sq['T'].values),alpha=0.25)
plt.show()
    
# *** Survival is random w.r.t. X input values ***
    
## run a set of CoxPH fits, generate a boxplot of the outcomes
c_index_list = []
for j in range(100):
    # potential X value, but unrelated in this case
    x = np.random.poisson(lam=10,size=3000)

    toy_dataset = gen_survival(x,x_relation=None)

    cph = CoxPHFitter()
    cph.fit(toy_dataset,duration_col='T',event_col='event_obs',show_progress=False)
    c_index_list.append(cph.score_)



# *** Survival is _linear_ w.r.t. X input values ***
    
c_index_list_lin = []
for j in range(100):
    # potential X value, but unrelated in this case
    x = np.random.poisson(lam=10,size=3000)

    toy_dataset = gen_survival(x,x_relation='lin')

    cph = CoxPHFitter()
    cph.fit(toy_dataset,duration_col='T',event_col='event_obs',show_progress=False)
    c_index_list_lin.append(cph.score_)

# *** Survival is _sq_ w.r.t. X input values ***
    
c_index_list_sq = []
for j in range(100):
    # potential X value, but unrelated in this case
    x = np.random.poisson(lam=10,size=3000)

    toy_dataset = gen_survival(x,x_relation='sq')

    cph = CoxPHFitter()
    cph.fit(toy_dataset,duration_col='T',event_col='event_obs',show_progress=False)
    c_index_list_sq.append(cph.score_)
    
# boxplot of c_index's
print("random",np.array(c_index_list).mean())
print("linear",np.array(c_index_list_lin).mean())
print("square",np.array(c_index_list_sq).mean())
plt.boxplot(np.array(c_index_list))
plt.boxplot(np.array(c_index_list_lin))
plt.boxplot(np.array(c_index_list_sq))
plt.show()
ipdb.set_trace()