'''
Simple MTLR example from: https://arxiv.org/pdf/1801.05512.pdf

x_1 ~ exp(lambda=0.1)
x_2 ~ Normal(mu=10,sigma^2=5)
x_3 ~ Poisson(lambda=5)

Generate 3,000 data points of x sampled from this distribution

** adjust time to get 40% of actual events ?? **

Create y value 

non-linear example:

T ~ Weibull(lambda=0.1*nu(X),p=2.1)
where nu(X) = (-0.5*x_1 + 9*x_2 + 19*x_3)^2

'''
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(20180531) #for reproducability

import pandas as pd

import ipdb

x_1 = np.random.exponential(scale=1./0.1,size=3000)
x_2 = np.random.normal(loc=10,scale=np.sqrt(5),size=3000)
x_3 = np.random.poisson(lam=5,size=3000)

# square 
nu = np.power( (-0.5*x_1 + 9.*x_2 + 19.*x_3) ,2)
# #gaussian
# nu = np.exp(-0.5* (-0.5*x_1 + 9.*x_2 + 19.*x_3))

## np returns 1-param weibull, see: doc for 2 param 
T = (0.1 * nu) * np.random.weibull(a=2.1,size=3000)

## pick censor level to get 40% of actual events
censor_level = np.percentile(T,40)

censored_T = T.clip(0,censor_level)
event_observed = np.ones_like(T)
event_observed[T>censor_level] = 0



#lifelines CoxPH model
toy_dataset = pd.DataFrame([x_1,x_2,x_3,censored_T,event_observed]).transpose()
toy_dataset.columns = ['x_1','x_2','x_3','T','event_obs']


from lifelines import CoxPHFitter

cph = CoxPHFitter()
## step_size here matters, otherwise you end up with convergence halted
cph.fit(toy_dataset,duration_col='T',event_col='event_obs',show_progress=True)#,step_size=0.25)
cph.print_summary()

ipdb.set_trace()