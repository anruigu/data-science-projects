#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import plotly.graph_objects as go
import streamlit as st


# ### Loading in the Data
#
# First, we read in four COVID-19-related .csv files using the pandas library (data as of 7/18/2020). Additionally, we read in a table of state abbreviations that we will use in later analysis as well.

# In[10]:


states = pd.read_csv('7.17states.csv')
counties = pd.read_csv('county_data_abridged.csv')
cases = pd.read_csv('1time_series_covid19_confirmed_US.csv')
deaths = pd.read_csv('1time_series_covid19_deaths_US.csv')
state_abbrev = pd.read_csv('state_abbreviations.csv')

pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', None)


# In[11]:


st.title('COVID-19 Predictor')


# In[12]:
statelist = list(cases['Province_State'].unique())
statelist.remove('American Samoa')
statelist.remove('Northern Mariana Islands')
statelist.remove('Diamond Princess')
statelist.remove('Grand Princess')
selected_state = st.selectbox('Which state are you in?',statelist)

# Work with confirmed cases only in California counties
caliConfirmed = cases[cases['Province_State']==selected_state]

# Filter out unncessary rows
caliConfirmed = caliConfirmed[caliConfirmed['Lat'] != 0]

# Accumulated cases every day
calitotal = caliConfirmed.groupby(['Province_State']).sum().iloc[:,10:]
calitotalTranspose = calitotal.transpose()


# State-wide lockdown occured on March 19. We filtered out columns that contained data from before the lockdown date, as prior to that day, the vast majority of people were not taking drastic enforced prevention measures. After the shelter-in-place order was put into effect, more people were taking similar prevention measures and the proportion of people susceptible accordingly decreased.
#

# In[13]:


lockdowncol = calitotal.columns.get_loc("3/19/20")
caliAfterLockdown = calitotal.groupby(['Province_State']).sum().iloc[:,lockdowncol:]
caliAfterLockdownTranspose = caliAfterLockdown.transpose()


# In[14]:


import matplotlib.pyplot as plt

Day = caliAfterLockdownTranspose.iloc[:,0].index
AccumCases = caliAfterLockdownTranspose.iloc[:,0].values
plt.figure(figsize = (25,8))
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Confirmed Cases in California Since Lockdown");

plt.plot(np.linspace(0, len(AccumCases), len(AccumCases)),AccumCases);
new_xticks = [Day[0], Day[9], Day[19], Day[29], Day[39], Day[49], Day[59], Day[69], Day[79], Day[89], Day[99], Day[109], Day[119]]
plt.xticks(np.arange(0,121,10),new_xticks);
plt.show()


# We estimate the current population of California as the sum of population estimates of each county in 2018.

# In[15]:


# Total population of California, as given by the state population estimate in 2018.
sumCounties = counties[counties['StateName'] == 'CA'].groupby(['StateName']).sum()
population_num = sumCounties.columns.get_loc("PopulationEstimate2018")
cali_Population = sumCounties.iloc[:,population_num].values[0]


# To model the spread of COVID-19, we used the SIR model, which is often used in modeling the spread of pandemics. The SIR model is based on a series of differential equations, but we instead used an iterative approximation to the solutions for sake of simplicity. This model is guided by three variables: the population susceptible to disease, the poulation currently infected, and the population recovered from the disease.

# In[16]:


# Recovery
def recovery(i, r, s, t, beta, gamma, N):
    value = r + gamma*i
    return value


# In[17]:


# Infected
def infected(i, r, s, t, beta, gamma, N):
    value = i + beta*i*s/N - gamma*i
    return value


# In[18]:


# Susceptible
def susceptible(i, r, s, t, beta, gamma, N):
    value = s - beta*i*s/N
    return value


# SIR bases its modeling off these three components, but some parameters also affect its behavior, such as total population, number of days someone can carry and spread the disease, and how many people can get infected by one infected person in a day. Since shelter-in-place is enforced, we do not expect the entire population to be susceptible to the virus, so we multiply the total population of California by a coefficient of susceptibility (less than 1). We'll say because of lockdown precautions, only 2 out of 6 Californians are susceptible to the disease.

# In[19]:


# Iterative derivation of SIR modeling over time
def sir(beta, gamma, N, s, i, r):
    t = 0
    s_new = susceptible(i, r, s, t, beta, gamma, N)
    i_new = infected(i, r, s, t, beta, gamma, N)
    r_new = recovery(i, r, s, t, beta, gamma, N)
    return [s_new, i_new, r_new]

susceptable_rate = st.slider('How susceptible is your state?',0,60,30)
susceptable_rate = susceptable_rate/100

initial_cases = AccumCases[0]

# Initial counts of susceptible, initial, and recovered counts in California.
s = cali_Population*susceptable_rate - initial_cases
i= initial_cases
r = 0

# Total population in the state.
n = cali_Population

# Days since shelter in place has started until 7/17/20, where our recorded data ends.
days = 121


# We create an SIR model over the given number of days and find the optimal values for the parameters beta and gamma, which are expected number of people an infected person infects in a day and the rate of recovery, respectively. To determine the accuracy of parameters, we look at the paramaters with minimal error from the actual data calculated using the root mean squared error.

# In[20]:


from scipy.optimize import minimize

# Change in each component (per day)
def sir(beta, gamma, N, s, i, r):
    t = 0
    s_new = susceptible(i, r, s, t, beta, gamma, N)
    i_new = infected(i, r, s, t, beta, gamma, N)
    r_new = recovery(i, r, s, t, beta, gamma, N)
    return [s_new, i_new, r_new]

# Return an iteration of how each component changes over the given
# length of time.
def sirModel(b, g, n, s, i, r, days):
    susceptible_array = []
    recovery_array = []
    infected_array = []

    susceptible_array.append(s)
    infected_array.append(i)
    recovery_array.append(r)

    for t in range(days - 1):
        SIR_vals = sir(b,g,n,s,i,r)

        susceptible_array.append(SIR_vals[0])
        infected_array.append(SIR_vals[1])
        recovery_array.append(SIR_vals[2])
        s = SIR_vals[0]
        i = SIR_vals[1]
        r = SIR_vals[2]

    return (susceptible_array, infected_array, recovery_array)

# Optimizes beta and gamma parameters
def loss(point, data):
    susceptible_array = []
    recovery_array = []
    infected_array = []
    b, g = point
    s = cali_Population * susceptable_rate - initial_cases
    i = initial_cases
    r = 0
    n = cali_Population

    sir_data = sirModel(b, g, n, s, i, r, len(data))

    susceptible_array = sir_data[0]
    infected_array = sir_data[1]
    recovery_array = sir_data[2]
    total = np.add(infected_array, recovery_array)
    rmse = np.sqrt(np.mean((total - data)**2))
    return rmse


# We optimize the loss defined by the root mean squared error with `scipy`'s `optimize`.

# In[21]:


data = AccumCases
optimal = minimize(
    loss,
    [0.01, 0.01],
    args=(data),
    method='L-BFGS-B',
    bounds=[(0.000001, 5), (0.000001, 1)]
)
beta, gamma = optimal.x
prediction = sirModel(beta, gamma, n, s, i, r, days)
print(beta,gamma)


# We now look at how each component modeled by SIR with the new optimal parameters changes over time.

# In[22]:


plt.figure(figsize = (25,8))
plt.xlabel("Date")
plt.plot(np.linspace(0,days,days),prediction[1], label = 'infected')
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Infected California Population Over Time");
new_xticks = [Day[0], Day[9],Day[19], Day[29], Day[39], Day[49], Day[59], Day[69], Day[79], Day[89], Day[99], Day[109], Day[119]]
plt.xticks(np.arange(0,121,10),new_xticks);


# In[23]:


plt.figure(figsize = (25,8))
plt.plot(np.linspace(0,days,days),prediction[2], label = 'recovery');
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Recovered California Population Over Time");
new_xticks = [Day[0], Day[9], Day[19], Day[29], Day[39], Day[49], Day[59], Day[69], Day[79], Day[89], Day[99], Day[109], Day[119]]
plt.xticks(np.arange(0,121,10),new_xticks);


# In[24]:


plt.figure(figsize = (25,8))
plt.plot(np.linspace(0,days,days),prediction[0], label = 'susceptible');
plt.xlabel("Date")
plt.ylabel("Count")
plt.title("Susceptible (to Illness) California Population Over Time");
new_xticks = [Day[0], Day[9], Day[19], Day[29], Day[39], Day[49], Day[59], Day[69], Day[79], Day[89], Day[99], Day[109], Day[119]]
plt.xticks(np.arange(0,121,10),new_xticks);


# SIR model usually models all three components on the same graph, but the amount of susceptible people in this model is extremely large relative to the other two components. If they were all plotted on the same graph, the y-axis would be so large that the behavior of recovered and infected cases would appear almost zero. Since the susceptible population is not the focus of this question, we looked at just the recovered and infected predictions to more clearly see their behavior and how well they predict compared to the actual data.

# In[25]:


s = cali_Population* susceptable_rate - initial_cases
i = initial_cases
r = 0
n = cali_Population
days = 300
predicted_trend = sirModel(beta, gamma, n, s, i, r, days)
plt.plot(np.linspace(0,days,days),predicted_trend[1], label = 'infected');
plt.plot(np.linspace(0,days,days),predicted_trend[2], label = 'recovered');
plt.xlabel("Date Since Lockdown")
plt.ylabel("Count")
plt.title("Susceptible and Infected California Population Over Time");
plt.legend();


# Using these optimized beta and gamma values, we can predict the confirmed cases using SIR. The confirmed cases are the sum of infected and recovered cases.

# In[26]:


accum_predicted_cases = np.add(predicted_trend[1], predicted_trend[2])
plt.figure(figsize=(20,8))
plt.plot(np.linspace(0,days,days), accum_predicted_cases, label = 'predicted');
plt.plot(np.linspace(0,121,121), AccumCases, label = 'actual');
plt.xlabel("Date Since Lockdown")
plt.ylabel("Count")
plt.title("Predicting cases in California since Lockdown");
plt.legend();


# The model predicts the current trend of cases extremely accurately! We can also look at how our model might predict new cases per day.

# In[27]:


accum_predicted_cases_Yesterday_Series = pd.Series(accum_predicted_cases)
accum_predicted_cases_Today_Series = pd.Series(accum_predicted_cases[1:])
accum_predicted_cases_daily = accum_predicted_cases_Today_Series - accum_predicted_cases_Yesterday_Series

# Accumulated new cases every day by subtracting current accumulated cases today
# from yesterday's cases.
today_lockdown = np.concatenate(caliAfterLockdownTranspose.values[1:])
confirmedCasesToday = pd.Series(today_lockdown);
confirmedCasesYesterday = pd.Series(np.concatenate(caliAfterLockdownTranspose.values))
confirmedNew = (confirmedCasesToday - confirmedCasesYesterday)

newCaliCasesDaily = pd.DataFrame(confirmedNew)
plt.figure(figsize=(20,8))
newCaliCaseAvg = newCaliCasesDaily.rolling(7).mean()
plt.plot(newCaliCaseAvg[0].index,newCaliCaseAvg[0].values, label = 'Actual (with Rolling Average)');
plt.plot(np.linspace(0,days, days), accum_predicted_cases_daily, label = 'Predicted');
plt.xlabel("Date")
plt.ylabel("New Cases Reported")
plt.title("Predicting new cases in California since Lockdown");
plt.legend();
plt.show()


# This model predicts that we are still on the upward trend of new cases rising daily. However, the trend of new cases seems to slow down at the ~150 days after Lockdown mark. It will be interesting to see if this prediction is true in the future!

# We can also look at the effect of increasing the susceptible constant: the effect when more people (that are currently healthy) come into contact with the disease.

# In[28]:


# What if two out of five Californians are suspectible to this disease?

i = initial_cases
r = 0
days = 200
optimal = minimize(
    loss,
    [0.01, 0.01],
    args=(data),
    method='L-BFGS-B',
    bounds=[(0.000001, 5), (0.000001, 1)]
)
beta, gamma = optimal.x
prediction = sirModel(beta, gamma, n, s, i, r, days)
print(beta,gamma)

prediction_Larger_Susceptible = sirModel(beta, gamma, n, s, i, r, days)


# In[29]:


accum_predicted_cases = np.add(prediction_Larger_Susceptible[1], prediction_Larger_Susceptible[2])
fig = plt.figure(figsize=(20,8))
plt.plot(np.linspace(0,days,days),accum_predicted_cases, label = 'predicted (more susceptible)');
plt.plot(np.linspace(0,121,121),AccumCases, label = 'actual');
plt.xlabel("Date Since Lockdown")
plt.ylabel("Count")
plt.title("Predicting cases in California since Lockdown");
plt.legend();
st.plotly_chart(fig)


# We see that this new model (susceptible coefficient of .4) also predicts the current trend of cases accurately. However, as time goes on, the cases will grow faster than in the previous model (susceptible coefficient of .2); after 200 days, this graph predicts there will be > 1 million total confirmed cases while a susceptible coefficient of .2 predicts around 1 million confirmed cases.

# In[ ]:
