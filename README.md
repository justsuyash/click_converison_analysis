```python
import pandas as pd
#read from google drive
data = pd.read_csv('https://drive.google.com/uc?export=download&id=1H_-mi6NzeZt1PhcanHofcf3H6UiItmxO')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>source</th>
      <th>device</th>
      <th>operative_system</th>
      <th>lat</th>
      <th>long</th>
      <th>weekday</th>
      <th>yearweek</th>
      <th>converted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>seo_facebook</td>
      <td>web</td>
      <td>mac</td>
      <td>38.89</td>
      <td>-94.81</td>
      <td>Friday</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>seo-google</td>
      <td>mobile</td>
      <td>android</td>
      <td>41.68</td>
      <td>-72.94</td>
      <td>Friday</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>friend_referral</td>
      <td>mobile</td>
      <td>iOS</td>
      <td>39.74</td>
      <td>-75.53</td>
      <td>Saturday</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>ads-google</td>
      <td>mobile</td>
      <td>android</td>
      <td>37.99</td>
      <td>-121.80</td>
      <td>Friday</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>ads-google</td>
      <td>mobile</td>
      <td>android</td>
      <td>41.08</td>
      <td>-81.52</td>
      <td>Wednesday</td>
      <td>14</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.device.value_counts()
```




    mobile    162321
    web       113295
    Name: device, dtype: int64




```python
data.operative_system.value_counts()
```




    windows    87721
    iOS        82982
    android    65305
    mac        21831
    other      14143
    linux       3634
    Name: operative_system, dtype: int64




```python
data.weekday.value_counts()
```




    Friday       80047
    Saturday     64632
    Sunday       48512
    Thursday     32187
    Tuesday      17104
    Monday       16722
    Wednesday    16412
    Name: weekday, dtype: int64




```python
data.source.value_counts()
```




    direct_traffic     52594
    ads-google         51576
    ads_facebook       46365
    ads_other          26084
    seo-google         20157
    ads-bing           19887
    seo_facebook       18473
    friend_referral    18011
    seo-other           8058
    ads-yahoo           6576
    seo-yahoo           5961
    seo-bing            1874
    Name: source, dtype: int64




```python
data.yearweek.value_counts()
```




    15    21779
    10    21489
    19    21463
    12    21332
    13    21247
    18    21215
    22    21134
    11    21081
    16    21037
    17    21034
    20    21021
    14    20895
    21    20889
    Name: yearweek, dtype: int64




```python
data.converted.value_counts()
```




    0    270597
    1      5019
    Name: converted, dtype: int64




```python
converted_adsbysource = data.groupby('source')[['source','converted']].sum('converted').sort_values(by='converted',ascending=False)
```


```python
data.groupby('source')[['source','converted']].sum('converted').sort_values(by='converted',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>converted</th>
    </tr>
    <tr>
      <th>source</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ads-google</th>
      <td>1102</td>
    </tr>
    <tr>
      <th>ads_facebook</th>
      <td>983</td>
    </tr>
    <tr>
      <th>friend_referral</th>
      <td>683</td>
    </tr>
    <tr>
      <th>direct_traffic</th>
      <td>639</td>
    </tr>
    <tr>
      <th>ads_other</th>
      <td>375</td>
    </tr>
    <tr>
      <th>seo-google</th>
      <td>343</td>
    </tr>
    <tr>
      <th>seo_facebook</th>
      <td>293</td>
    </tr>
    <tr>
      <th>ads-bing</th>
      <td>238</td>
    </tr>
    <tr>
      <th>seo-other</th>
      <td>128</td>
    </tr>
    <tr>
      <th>seo-yahoo</th>
      <td>99</td>
    </tr>
    <tr>
      <th>ads-yahoo</th>
      <td>96</td>
    </tr>
    <tr>
      <th>seo-bing</th>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing


```python
data_dummy = pd.get_dummies(data,drop_first=True)
```


```python
data_dummy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>lat</th>
      <th>long</th>
      <th>yearweek</th>
      <th>converted</th>
      <th>source_ads-google</th>
      <th>source_ads-yahoo</th>
      <th>source_ads_facebook</th>
      <th>source_ads_other</th>
      <th>source_direct_traffic</th>
      <th>...</th>
      <th>operative_system_linux</th>
      <th>operative_system_mac</th>
      <th>operative_system_other</th>
      <th>operative_system_windows</th>
      <th>weekday_Monday</th>
      <th>weekday_Saturday</th>
      <th>weekday_Sunday</th>
      <th>weekday_Thursday</th>
      <th>weekday_Tuesday</th>
      <th>weekday_Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>38.89</td>
      <td>-94.81</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>41.68</td>
      <td>-72.94</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14</td>
      <td>39.74</td>
      <td>-75.53</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>37.99</td>
      <td>-121.80</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>41.08</td>
      <td>-81.52</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



## Building a logistic Regression


```python
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
```


```python
train = data_dummy.drop('converted',axis=1)
```


```python
train['intercept'] = 1 
```


```python
logit = sm.Logit(data_dummy['converted'],train)
```


```python
output = logit.fit()
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.089555
             Iterations: 35
    

    C:\Users\aratr\anaconda3\lib\site-packages\statsmodels\base\model.py:566: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      warnings.warn("Maximum Likelihood optimization failed to "
    


```python
output_df = pd.DataFrame(dict(coeff=output.params,se=output.bse,pvalue=output.pvalues,t_Value=output.tvalues))
```


```python
output_df[output_df['pvalue']<0.05].sort_values(by='coeff',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coeff</th>
      <th>se</th>
      <th>pvalue</th>
      <th>t_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>source_friend_referral</th>
      <td>1.177281</td>
      <td>0.076029</td>
      <td>4.409649e-54</td>
      <td>15.484575</td>
    </tr>
    <tr>
      <th>source_ads-google</th>
      <td>0.587031</td>
      <td>0.071997</td>
      <td>3.533160e-16</td>
      <td>8.153577</td>
    </tr>
    <tr>
      <th>source_ads_facebook</th>
      <td>0.577365</td>
      <td>0.072771</td>
      <td>2.121665e-15</td>
      <td>7.934018</td>
    </tr>
    <tr>
      <th>source_seo-bing</th>
      <td>0.573800</td>
      <td>0.172699</td>
      <td>8.920106e-04</td>
      <td>3.322543</td>
    </tr>
    <tr>
      <th>operative_system_iOS</th>
      <td>0.407536</td>
      <td>0.040065</td>
      <td>2.646118e-24</td>
      <td>10.171930</td>
    </tr>
    <tr>
      <th>source_seo-google</th>
      <td>0.356743</td>
      <td>0.084992</td>
      <td>2.700258e-05</td>
      <td>4.197376</td>
    </tr>
    <tr>
      <th>source_seo-yahoo</th>
      <td>0.336044</td>
      <td>0.120564</td>
      <td>5.315571e-03</td>
      <td>2.787261</td>
    </tr>
    <tr>
      <th>source_seo_facebook</th>
      <td>0.290116</td>
      <td>0.087898</td>
      <td>9.647307e-04</td>
      <td>3.300615</td>
    </tr>
    <tr>
      <th>source_seo-other</th>
      <td>0.287583</td>
      <td>0.110455</td>
      <td>9.224834e-03</td>
      <td>2.603607</td>
    </tr>
    <tr>
      <th>source_ads_other</th>
      <td>0.186544</td>
      <td>0.083446</td>
      <td>2.538346e-02</td>
      <td>2.235517</td>
    </tr>
    <tr>
      <th>weekday_Tuesday</th>
      <td>-0.142619</td>
      <td>0.066179</td>
      <td>3.115732e-02</td>
      <td>-2.155058</td>
    </tr>
    <tr>
      <th>operative_system_other</th>
      <td>-0.170095</td>
      <td>0.082942</td>
      <td>4.028980e-02</td>
      <td>-2.050765</td>
    </tr>
    <tr>
      <th>intercept</th>
      <td>-4.409939</td>
      <td>0.168811</td>
      <td>1.967960e-150</td>
      <td>-26.123578</td>
    </tr>
  </tbody>
</table>
</div>




```python
converted_adsbysource.reset_index(inplace=True)
```


```python
source_count = pd.DataFrame(data.source.value_counts())
```


```python
source_count.reset_index(inplace=True)
```


```python
percentage_conversion = pd.merge(converted_adsbysource,source_count,left_on='source',right_on='index')
```


```python
percentage_conversion['perc'] = percentage_conversion['converted']/percentage_conversion['source_y']
```


```python
percentage_conversion.sort_values('perc',ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_x</th>
      <th>converted</th>
      <th>index</th>
      <th>source_y</th>
      <th>perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>friend_referral</td>
      <td>683</td>
      <td>friend_referral</td>
      <td>18011</td>
      <td>0.037921</td>
    </tr>
    <tr>
      <th>0</th>
      <td>ads-google</td>
      <td>1102</td>
      <td>ads-google</td>
      <td>51576</td>
      <td>0.021367</td>
    </tr>
    <tr>
      <th>11</th>
      <td>seo-bing</td>
      <td>40</td>
      <td>seo-bing</td>
      <td>1874</td>
      <td>0.021345</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ads_facebook</td>
      <td>983</td>
      <td>ads_facebook</td>
      <td>46365</td>
      <td>0.021201</td>
    </tr>
    <tr>
      <th>5</th>
      <td>seo-google</td>
      <td>343</td>
      <td>seo-google</td>
      <td>20157</td>
      <td>0.017016</td>
    </tr>
    <tr>
      <th>9</th>
      <td>seo-yahoo</td>
      <td>99</td>
      <td>seo-yahoo</td>
      <td>5961</td>
      <td>0.016608</td>
    </tr>
    <tr>
      <th>8</th>
      <td>seo-other</td>
      <td>128</td>
      <td>seo-other</td>
      <td>8058</td>
      <td>0.015885</td>
    </tr>
    <tr>
      <th>6</th>
      <td>seo_facebook</td>
      <td>293</td>
      <td>seo_facebook</td>
      <td>18473</td>
      <td>0.015861</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ads-yahoo</td>
      <td>96</td>
      <td>ads-yahoo</td>
      <td>6576</td>
      <td>0.014599</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ads_other</td>
      <td>375</td>
      <td>ads_other</td>
      <td>26084</td>
      <td>0.014377</td>
    </tr>
    <tr>
      <th>3</th>
      <td>direct_traffic</td>
      <td>639</td>
      <td>direct_traffic</td>
      <td>52594</td>
      <td>0.012150</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ads-bing</td>
      <td>238</td>
      <td>ads-bing</td>
      <td>19887</td>
      <td>0.011968</td>
    </tr>
  </tbody>
</table>
</div>



## Insights: 

#### 1. Friend Referrals are the way to go, we should talk to the manager regarding what kind of efforts are being taken in this direction and possibly spend more money in this program, if its justified by the Customer Lifetime Value.

#### 2. Coefficients of ADS by Google, Facebook and Bing are almost the same, so is the percentage of conversion however we are sending out a lot of "Other" ads we would like to check the spending on those ads, also on bing ads the conversion rate is too low, we maybe need to target the users in a different way.

#### 3. We see that iOS users are more likely to click on the ad, Could we compare the positioning, placement and UI for the ad? Maybe we could redesign the Windows/Android App.

#### 4. It seems like the days dont matter, but Tuesday is significantly worse than Friday.


```python

```

## Creating a tree to check for additional Rules :


```python
from sklearn.tree import DecisionTreeClassifier
from graphviz import Source
from sklearn.tree import export_graphviz
```


```python
tree = DecisionTreeClassifier(max_depth=6,class_weight='balanced',min_impurity_decrease=0.001)
```


```python
tree.fit(train,data_dummy['converted'])
```




    DecisionTreeClassifier(class_weight='balanced', max_depth=6,
                           min_impurity_decrease=0.001)




```python
export_graphviz(tree,out_file="ex_tree.dot",feature_names=train.columns,proportion=True,rotate=True)
```


```python
S = Source.from_file("ex_tree.dot")
```


```python
S.view()
```




    'ex_tree.dot.pdf'



## Analysis

### 1. We see that as we guessed "Refered by a friend" is very important here as well and as we go down that is "Refer Friend" is true the probabllity  of correct prediction is incerased to 68%, Everything becomes irrelevant as well. Thus we should foucus on this area by both of our analysis

### 2. When a person directly visits a website without a referral, we see that only 30% people converted, we need to probably check in with the manager if the UI/UX is upto the mark.

### 3. Anything after that doesnt give us much information


```python

```

## We already have a couple of insights by now, I will check on PDP


```python
import pdp
```


```python
feat_original = data_dummy.columns.drop('converted')
```


```python
pdp.pdp_
```


```python
for i in range(len(feat_original)):
    
    plot_variable = [e for e in train.columns if e.startswith(feat_original[i])]
   
    if len(plot_variable) == 1:
        pdp_iso = pdp.pdp_isolate( model=rf, dataset=train.columns, model_features=train.columns, feature=plot_variable[0], num_grid_points=50)
        pdp_dataset = pandas.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
        #pdpbox has several options if you want to use their built-in plots. I personally prefer just using .plot. It is totally subjective obviously.
        pdp_dataset.plot(title=feat_original[i])
        plt.show()
         
    #categorical variables with several levels
    else:
        pdp_iso = pdp.pdp_isolate( model=rf, dataset=train.columns, model_features=train.columns,  feature=plot_variable, num_grid_points=50)
        pdp_dataset = pandas.Series(pdp_iso.pdp, index=pdp_iso.display_columns)
        pdp_dataset.sort_values(ascending=False).plot(kind='bar', title=feat_original[i])
        plt.show()
    plt.close()   
    
    
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-136-f6f2aff8eca6> in <module>
          4 
          5     if len(plot_variable) == 1:
    ----> 6         pdp_iso = pdp.pdp_isolate( model=rf, dataset=train.columns, model_features=train.columns, feature=plot_variable[0], num_grid_points=50)
          7         pdp_dataset = pandas.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
          8         #pdpbox has several options if you want to use their built-in plots. I personally prefer just using .plot. It is totally subjective obviously.
    

    AttributeError: module 'pdp' has no attribute 'pdp_isolate'



```python
# Matplot Lib WHeel Error while installing pdpbox, still trying to fix it 
```

# My Top 3 Insights

### 1. Imporve referral rewards, introduce some kind of gamification where users can collect more points together

### 2. Improve UI/UX design, Make Better Funnel for people visiting the website

### 3. Target more iOS users, talk with the manager on why Windows/Android ads are not doing better

### 4. Saves costs on "other" types of ad


```python

```
