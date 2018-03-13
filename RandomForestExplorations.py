import numpy as np
import pandas as pd
import pickle as pk
import heapq

def moving_average(a, n=10) :
    summed = np.cumsum(a,dtype=float)
    #resultant vector always has length N-(n-1)
    summed[n:] = summed[n:] - summed[:-n]
    return summed[n - 1:] / n

def labeller(price,n_interval,n_test):
    mean_and_stdev = np.array(list(
                   map(lambda n : [np.mean(price[n+n_interval:n+n_interval+n_test])
                                   ,np.std(price[n+n_interval:n+n_interval+n_test])]
                       ,np.arange(len(price)-n_interval-n_test+1))))
    test_prices = price[:-n_interval-n_test+1]
    labels = np.zeros(len(test_prices),dtype=int)
    labels[test_prices > mean_and_stdev[:,0]+ 2* mean_and_stdev[:,1]] = -1
    labels[test_prices < mean_and_stdev[:,0]- 2* mean_and_stdev[:,1]] = 1
    return labels

#load in orderbook data
with open('/Users/colint/Documents/gemini_data/ethusd.pkl', 'rb') as ethusdFile:
    ethusdList = pk.load(ethusdFile)

#process into usable form:
cf = lambda x: dt.strptime(x,'%y-%m-%d-%H.%M.%S.%f')

times = [cf(x['timestamp']) for x in ethusdList]

eu_asks = []
eu_bids = []
for x in ethusdList:
    lowest_asks = heapq.nsmallest(5, list(x['asks'].keys()))
    highest_bids = heapq.nlargest(5,list(x['bids'].keys()))
    bid_sorted = np.sort(list(x['bids'].keys()))
    eu_asks.append([[la, x['asks'][la]] for la in lowest_asks] )
    eu_bids.append([[hb, x['bids'][hb]] for hb in highest_bids] )

askM = np.array(eu_asks)
bidM = np.array(eu_bids)

#raw values
mid_price = (askM[:,0,0] + bidM[:,0,0])/2
aVsum = np.sum(askM[:,:,1],axis=1)
bVsum = np.sum(bidM[:,:,1],axis=1)
v_imbalance = (aVsum - bVsum)/(aVsum + bVsum)
micro_price = (askM[:,0,0] * askM[:,0,1] + bidM[:,0,0] * bidM[:,0,1])/(askM[:,0,1]+bidM[:,0,1])
q_spread = askM[:,0,0] - bidM[:,0,0]

#moving averages
vI_m10 = moving_average(v_imbalance,n=10)
midP_m10 = moving_average(mid_price,n=10)
microP_m10 = moving_average(micro_price,n=10)
spread_m10 = moving_average(q_spread,n=10)

midP_m50 = moving_average(mid_price,n=50)

vI_m100 = moving_average(v_imbalance,n=100)
midP_m100 = moving_average(mid_price,n=100)
microP_m100 = moving_average(micro_price,n=100)

#label by looking at the 50 sec mean price and stdev, 2 mins from current time.
labels = labeller(mid_price,240,100)

#put everything into a dataframe.
df = pd.DataFrame(np.vstack([np.array(times)[99:-240-100+1],mid_price[99:-240-100+1],labels[99:],
    vI_m10[90:-240-100+1],midP_m10[90:-240-100+1],
    microP_m10[90:-240-100+1],spread_m10[90:-240-100+1],
    midP_m50[50:-240-100+1],vI_m100[:-240-100+1],
    midP_m100[:-240-100+1],microP_m100[:-240-100+1]]).T,
  columns=['Time','Price','Label','VImbalance10','Price10','mPrice10','Spread10','Price50'
   ,'VImbalance100','Price100','mPrice100'])

#quick and dirty choice of 80% of data
df['Test'] = pd.Series(np.random.randn(len(df)) < 0.85, index=df.index)
training_set,validation_set = df[df['Test']], df[~df['Test']]
features = df.columns[[1,3,4,5,6,7,8,9,10]]

#initialize random forest and start fit.
clf = RandomForestClassifier(max_depth=10, random_state=0)
labs = np.array(training_set['Label'],dtype=int)
clf.fit(training_set[features],labs)

#Test the predictions.
predictions = clf.predict(validation_set[features])
correct = np.sum(predictions==validation_set['Label'])
print("The predictions were correct {} out of {} times. ({}%)".format(correct,len(validation_set),correct/len(validation_set)*100))

#print out confusion matrix.
print(pd.crosstab(validation_set['Label'], predictions, rownames=['Actual actions'], colnames=['Predicted actions']))






