
import pandas as pd
import numpy as np
import holidays
from datetime import date
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn import metrics
from sklearn import calibration
from tqdm import tqdm
import matplotlib.pyplot as plt

data_flights=pd.read_csv("drive/MyDrive/flights_fixed.csv").drop(['Unnamed: 0'],axis=1)

data_flights.drop(["YEAR","TAIL_NUMBER","DEPARTURE_TIME","TAXI_OUT","WHEELS_OFF","SCHEDULED_TIME","ELAPSED_TIME",
                   "ELAPSED_TIME","DISTANCE","WHEELS_ON","TAXI_IN","CANCELLATION_REASON","AIR_SYSTEM_DELAY",
                  "SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY"],axis=1,inplace=True)

data_flights = data_flights[(data_flights.CANCELLED == 0) & (data_flights.DIVERTED == 0)]\
.drop(["DIVERTED","CANCELLED"],axis=1).dropna()

weekend_flights = data_flights[(data_flights.DAY_OF_WEEK == 6) | (data_flights.DAY_OF_WEEK == 7)]
weekday_flights = data_flights[(data_flights.DAY_OF_WEEK != 6) & (data_flights.DAY_OF_WEEK != 7)]

airports1= data_flights.groupby('ORIGIN_AIRPORT').count()['MONTH'].sort_values()
airports2 = data_flights.groupby('DESTINATION_AIRPORT').count()['MONTH'].sort_values()
valid_airports=set(airports1[airports1 >= airports1.quantile(q=0.9)].index)\
.intersection(set(airports1[airports2 >= airports2.quantile(q=0.9)].index))

data_flights_filtered = data_flights[data_flights.ORIGIN_AIRPORT.isin(valid_airports) & 
  data_flights.DESTINATION_AIRPORT.isin(valid_airports)]

airlines = data_flights_filtered.groupby('AIRLINE').count()['MONTH'].sort_values()
valid_airlines=set(airlines[airlines >= airlines.quantile(q=0.50)].index)

data_flights_filtered = data_flights_filtered[data_flights_filtered.AIRLINE.isin(valid_airlines)]

def is_holiday(month,day):
  date1 = date(2015,month,day)
  us_holi = set(holidays.US(years=2015).keys())
  return 1 if date1 in us_holi else 0

def get_season(month,day):
  if month <=2 or (month==3 and day <20) or (month ==12 and day>=21):
    return 'winter'
  if month < 6 or (month==6 and day<21):
    return "spring"
  if month<9 or (month==9 and day<23):
    return 'summer'
  return 'fall'

T='is_weekend'
Y= 'DEPARTURE_DELAY'
data_flights_filtered[T] = data_flights_filtered["DAY_OF_WEEK"]\
.apply(lambda x: 1 if x==6 or x==7 else 0)
data_flights_filtered['season'] = data_flights_filtered\
.apply(lambda x: get_season(x["MONTH"],x["DAY"]),axis=1)
data_flights_filtered['is_holiday'] = data_flights_filtered\
.apply(lambda x: is_holiday(x["MONTH"],x["DAY"]),axis=1)

covariates = ["ORIGIN_AIRPORT","DESTINATION_AIRPORT","AIRLINE","is_holiday",'season']
needed_data = data_flights_filtered[covariates+[T,Y]]
needed_data = pd.get_dummies(needed_data,drop_first=True)
needed_data.to_csv('processed.csv')

#sampling a sub sample of the data
needed_data = pd.read_csv('processed.csv')
needed_sample = needed_data.sample(n = 20000,random_state=100,replace=False)

def calculate_propensity(data,print_metrics=False,plot=False,inplace=False):
  Y = "DEPARTURE_DELAY"
  T = 'is_weekend'
  if not inplace:
    data_new = data.copy(deep=True)
  else:
    data_new =data
  learner = LogisticRegression()
  X = np.array(data_new.drop([Y, T],axis=1))
  T_arr = np.array(data_new[T])
  trained = learner.fit(X,T_arr)
  covariates = data_new.columns.tolist()
  covariates.remove(Y)
  covariates.remove(T)
  print('finished_training')
  data_new['propensity'] = data[covariates].apply(
    lambda s: trained.predict_proba(s.values[None])[0][1], axis=1
    )
  if print_metrics:
    print("Accuracy:",metrics.accuracy_score(T_arr, np.array(data[covariates].apply(
      lambda s: trained.predict(s.values[None])[0], axis=1)))) 

    print("Precision:",metrics.precision_score(T_arr, np.array(data[covariates].apply(
      lambda s: trained.predict(s.values[None])[0], axis=1)))) 

    print("Recall:",metrics.recall_score(T_arr, np.array(data[covariates].apply(
      lambda s: trained.predict(s.values[None])[0], axis=1)))) 
    print(trained.predict_proba(X)[:,1])
  if plot:
    true,pred = calibration.calibration_curve(T_arr,trained.predict_proba(X)[:,1],n_bins =10)
    sns.scatterplot(x= true,y=pred)
    sns.lineplot(x=np.arange(0,1.1,0.1),y=np.arange(0,1.1,0.1))
  if not inplace:
    return data_new

sample_prop = calculate_propensity(needed_sample)

sns.histplot(data =sample_prop,x='propensity',hue = 'is_weekend',bins =50,element='step')
plt.show()

#calculating ate with matching
def calc_ate_matching(data,k):
    data = data.copy(deep=True)
    T ="is_weekend"
    Y= 'DEPARTURE_DELAY'
    data['Y1'] =data.apply(lambda x: x[Y] if x[T] ==1 else np.nan,axis=1)
    data['Y0'] =data.apply(lambda x: x[Y] if x[T] ==0 else np.nan,axis=1)
    imputer = KNNImputer(n_neighbors=k)
    trained = imputer.fit(data)
    data[:] = trained.transform(data)
    return np.mean(np.array(data['Y1'])-np.array(data['Y0']))

def bootstrap(data,func,num_samples,level):
  np.random.seed(100)
  estimators = []
  n= len(data)
  for i in tqdm(range(num_samples)):
    sample =data.sample(n=n,replace=True)
    estimators.append(func(sample))
    CI = (np.quantile(estimators,q=(1-level)/2),np.quantile(estimators,q=1-((1-level)/2)))
  print(estimators)
  return np.std(estimators) , CI

#calculating ate using matching
matching_ate=[]
matching_se = []
matching_CI = []
for k in [1,3]:
  print(k)
  ate = calc_ate_matching(needed_sample,k)
  se,ci = bootstrap(needed_sample,lambda x: calc_ate_matching(x,k),200,0.95)
  matching_ate.append(ate)
  matching_se.append(se)
  matching_CI.append(ci)

def calc_ate_ipw(data):
  Y = 'DEPARTURE_DELAY'
  T='is_weekend'
  n = len(data)
  data_prop = calculate_propensity(data)
  data_prop['weighted1'] = data_prop[Y]/data_prop['propensity']
  data_prop['weighted0'] = data_prop[Y]/(1-data_prop['propensity'])

  y1 = data_prop[data_prop[T] ==1]['weighted1']
  y0 = data_prop[data_prop[T] ==0]['weighted0']
  return np.sum(y1)/n - np.sum(y0)/n

#returns att using s learner
def calc_ate_s_learner(data,learner):
  data = data.copy(deep=True)
  Y ='DEPARTURE_DELAY'
  T = 'is_weekend'
  X = np.array(data.drop([Y],axis=1))
  y = np.array(data[Y])
  trained = learner.fit(X,y)
  data.loc[:,T] =1
  X_treated_1 = np.array(data.drop([Y],axis=1))
  data.loc[:,T] =0
  X_treated_0 = np.array(data.drop([Y],axis=1))
  return np.mean(trained.predict(X_treated_1) - trained.predict(X_treated_0))

def t_learner (data,learner1,learner0):
  Y='DEPARTURE_DELAY'
  T='is_weekend'
  data =data.copy(deep=True)
  treated =   data[data[T] == 1]
  control =   data[data[T] == 0]
  X_treated = np.array(treated.drop([Y,T],axis=1))
  y_treated = np.array(treated[Y])
  X_control = np.array(control.drop([Y,T],axis=1))
  y_control = np.array(control[Y])
  trained1 = learner1.fit(X_treated,y_treated)
  trained0 = learner0.fit(X_control,y_control)
  X_1 = np.array(data.drop([Y,T],axis=1))
  X_0 = np.array(data.drop([Y,T],axis=1))
  return np.mean(trained1.predict(X_1) - trained0.predict(X_0))

def doubly_robust(data,learner1,learner0):
  
  Y='DEPARTURE_DELAY'
  T='is_weekend'
  n = len(data)
  data_prop = calculate_propensity(data)
  data = data.copy(deep=True)
  treated =   data[data[T] == 1]
  control =   data[data[T] == 0]
  X_treated = treated.drop([Y,T],axis=1)
  y_treated = treated[Y]
  X_control = control.drop([Y,T],axis=1)
  y_control = control[Y]
  trained1 = learner1.fit(X_treated,y_treated)
  trained0 = learner0.fit(X_control,y_control)
  X1 = data_prop.drop([T,Y,'propensity'],axis=1)
  X0 = data_prop.drop([T,Y,'propensity'],axis=1)
  data_prop['Y1_pred'] = trained1.predict(X1)
  data_prop['Y0_pred'] = trained0.predict(X0)
  data_prop['g1'] = data_prop['Y1_pred']+(data_prop[T]/data_prop['propensity'])\
  *(data_prop[Y]-data_prop['Y1_pred'])
  data_prop['g0'] = data_prop['Y0_pred']+((1-data_prop[T])/(1-data_prop['propensity']))\
  *(data_prop[Y]-data_prop['Y0_pred'])
  return np.mean(data_prop['g1']-data_prop['g0'])

lin_model =
ate_t_linear = s_learner(needed_sample,lin_model)
print(ate_s_linear)
se_s_linear,ci_s_linear = bootstrap(needed_sample,lambda x: s_learner(x,lin_model),100,0.95)

lin_model1 =LinearRegression()
lin_model0 =LinearRegression()
ate_t_linear = t_learner(needed_sample,lin_model1,lin_model0)
print(ate_t_linear)
se_t_linear,ci_t_linear = bootstrap(needed_sample,lambda x: t_learner(x,lin_model1,lin_model0),100,0.95)

lin_model1 =LinearRegression()
lin_model0 =LinearRegression()
ate_dr = doubly_robust(needed_sample,lin_model1,lin_model0)
print(ate_dr)
se_dr,ci_dr = bootstrap(needed_sample,lambda x: doubly_robust(x,lin_model1,lin_model0),100,0.95)

lin_model =LinearRegression()
ate_s_linear = calc_ate_s_learner(needed_sample,lin_model)
print(ate_s_linear)
se_s_linear,ci_s_linear = bootstrap(needed_sample,lambda x: calc_ate_s_learner(x,lin_model),100,0.95)

def select_season(data,season):
  if season == 'spring':
    return data[data.season_spring==1]
  if season == 'summer':
    return data[data.season_summer==1]
  if season == 'winter':
    return data[data.season_winter==1]
  return data[(data.season_spring==0) & (data.season_winter==0) & (data.season_summer==0)]

results={'s':{},'t':{},'ipw':{},'dr':{}}
for season in ['winter','summer','spring','fall']:
  data_copy = needed_sample.copy(deep=True)
  data_copy = select_season(data_copy,season)
  data_copy = data_copy.drop(['season_spring','season_winter','season_summer'],axis=1)
  lin_model =LinearRegression()
  ate_s_linear = calc_ate_s_learner(data_copy,lin_model)
  se_s_linear,ci_s_linear = bootstrap(data_copy,lambda x: calc_ate_s_learner(x,lin_model),100,0.95)
  results['s'][season] = {'ate':ate_s_linear,'se':se_s_linear,'ci':ci_s_linear}

for season in ['summer','spring','fall','winter',]:
  data_copy = needed_sample.copy(deep=True)
  data_copy = select_season(data_copy,season)
  data_copy = data_copy.drop(['season_spring','season_winter','season_summer'],axis=1)
  lin_model1 = LinearRegression()
  lin_model0 = LinearRegression()
  ate_t = t_learner(data_copy,lin_model1,lin_model0)
  se_t,ci_t = bootstrap(data_copy,lambda x: t_learner(x,lin_model1,lin_model0),100,0.95)
  results['t'][season] = {'ate':ate_t,'se':se_t,'ci':ci_t}

for season in ['summer','spring','fall','winter',]:
  data_copy = needed_sample.copy(deep=True)
  data_copy = select_season(data_copy,season)
  data_copy = data_copy.drop(['season_spring','season_winter','season_summer'],axis=1)
  lin_model1 = LinearRegression()
  lin_model0 = LinearRegression()
  ate_dr = doubly_robust(data_copy,lin_model1,lin_model0)
  se_dr,ci_dr = bootstrap(data_copy,lambda x: doubly_robust(x,lin_model1,lin_model0),100,0.95)
  results['dr'][season] = {'ate':ate_dr,'se':se_dr,'ci':ci_dr}

for season in ['summer','spring','fall','winter',]:
  data_copy = needed_sample.copy(deep=True)
  data_copy = select_season(data_copy,season)
  data_copy = data_copy.drop(['season_spring','season_winter','season_summer'],axis=1)
  ate = calc_ate_matching(data_copy,k=1)
  se,ci = bootstrap(data_copy,lambda x: calc_ate_matching(x,k=1),100,0.95)
  results['matching'][season] = {'ate':ate,'se':se,'ci':ci}

for season in ['summer','spring','fall','winter',]:
  data_copy = needed_sample.copy(deep=True)
  data_copy = select_season(data_copy,season)
  data_copy = data_copy.drop(['season_spring','season_winter','season_summer'],axis=1)
  ate = calc_ate_matching(data_copy,k=1)
  se_ipw,ci_ipw = bootstrap(data_copy,lambda x: calc_ate_matching(x,k=1),100,0.95)
  results['ipw'][season] = {'ate':ate_ipw,'se':se_ipw,'ci':ci_ipw}

def plot_ci(x,bottom,top,value):
  color = '#2187bb'
  left = x-0.25/2
  right = x+ 0.25/2
  plt.plot([x,x],[bottom,top],color=color)
  plt.plot([left,right],[bottom,bottom],color=color)
  plt.plot([left,right],[top,top],color=color)
  plt.plot(x,value,'o',color='#f44336')

plt.xticks([1,2,3,4,5],['s-learner','t_learner','ipw','matching','doubly robust'])
plot_ci(1,-2.04,0.527,-0.751)
plot_ci(2,-2.307,0.332,-0.902)
plot_ci(3,-2.273,0.332,-0.897)
plot_ci(4,-0.158,0.049,-0.01475)
plot_ci(5,-2.296,0.344,-0.885)
plt.title('confidence intervals for ATE on entire data')
plt.xlabel('method')
plt.show()

for season in ['summer','fall','spring','winter']:
  if season != 'winter':
    plt.xticks([1,2,3,4,5],['s-learner','t_learner','ipw','matching','doubly robust'])
    plot_ci(1,results['s'][season]['ci'][0],results['s'][season]['ci'][1],results['s'][season]['ate'])
    plot_ci(2,results['t'][season]['ci'][0],results['t'][season]['ci'][1],results['t'][season]['ate'])
    plot_ci(3,results['ipw'][season]['ci'][0],results['ipw'][season]['ci'][1],results['ipw'][season]['ate'])
    plot_ci(4,results['matching'][season]['ci'][0],
            results['matching'][season]['ci'][1],results['matching'][season]['ate'])
    plot_ci(5,results['dr'][season]['ci'][0],results['dr'][season]['ci'][1],results['dr'][season]['ate'])
    plt.title(f'confidence intervals for ATE on only on {season} data')
    plt.xlabel('method')
    plt.show()
  else:
    plt.xticks([1,2,3],['s-learner','ipw','matching'])
    plot_ci(1,results['s'][season]['ci'][0],results['s'][season]['ci'][1],results['s'][season]['ate'])
    plot_ci(2,results['ipw'][season]['ci'][0],results['ipw'][season]['ci'][1],results['ipw'][season]['ate'])
    plot_ci(3,results['matching'][season]['ci'][0],
            results['matching'][season]['ci'][1],results['matching'][season]['ate'])
    plt.title(f'confidence intervals for ATE on only on {season} data')
    plt.xlabel('method')
    plt.show()