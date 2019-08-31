## Heart Disease

Using examples from the [relevant Google course](https://developers.google.com/machine-learning/crash-course/ml-intro) `heart-disease.py` shows an effort to predict heart disease in patients using the [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
This repo should more or less be treated as an example of a ML algorithm not working :wink:

![?](https://i.kym-cdn.com/photos/images/newsfeed/000/234/765/b7e.jpg)

### Things going wrong
This example really doesn't have a lot going for itself, however it was a good first step.

* The sample data (300) is really not enough for predictions of this sort
* The features themselves are very specific and there is a lot of them
* Model overfits

### Example output
```
python3 heart-disease.py

********** Sample Data **********

     age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
32    44    1   1       130   219    0        0      188      0      0.0      2   0     2       1
113   43    1   0       110   211    0        1      161      0      0.0      2   0     3       1
221   55    1   0       140   217    0        1      111      1      5.6      0   0     3       0
139   64    1   0       128   263    0        1      105      1      0.2      1   1     3       1
151   71    0   0       112   149    0        1      125      0      1.6      1   0     2       1
..   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...    ...  ..   ...     ...
93    54    0   1       132   288    1        0      159      1      0.0      2   1     2       1
150   66    1   0       160   228    0        0      138      0      2.3      2   0     1       1
112   64    0   2       140   313    0        1      133      0      0.2      2   0     3       1
98    43    1   2       130   315    0        1      162      0      1.9      2   1     2       1
203   68    1   2       180   274    1        0      150      1      1.6      1   0     3       0

[303 rows x 14 columns]

********** Description **********

        age   sex    cp  trestbps  chol   fbs  restecg  thalach  exang  oldpeak  slope    ca  thal  target
count 303.0 303.0 303.0     303.0 303.0 303.0    303.0    303.0  303.0    303.0  303.0 303.0 303.0   303.0
mean   54.4   0.7   1.0     131.6 246.3   0.1      0.5    149.6    0.3      1.0    1.4   0.7   2.3     0.5
std     9.1   0.5   1.0      17.5  51.8   0.4      0.5     22.9    0.5      1.2    0.6   1.0   0.6     0.5
min    29.0   0.0   0.0      94.0 126.0   0.0      0.0     71.0    0.0      0.0    0.0   0.0   0.0     0.0
25%    47.5   0.0   0.0     120.0 211.0   0.0      0.0    133.5    0.0      0.0    1.0   0.0   2.0     0.0
50%    55.0   1.0   1.0     130.0 240.0   0.0      1.0    153.0    0.0      0.8    1.0   0.0   2.0     1.0
75%    61.0   1.0   2.0     140.0 274.5   0.0      1.0    166.0    1.0      1.6    2.0   1.0   3.0     1.0
max    77.0   1.0   3.0     200.0 564.0   1.0      2.0    202.0    1.0      6.2    2.0   4.0   3.0     1.0

********** Training Model **********

Period 0: 0.6843756228731918 0.691124163583721
Period 1: 0.6652756149318683 0.6719344953134975
Period 2: 0.6514210314225196 0.6579788043144807
Period 3: 0.6403290075102784 0.6467800838236305
Period 4: 0.6309682877789531 0.6373090153188931
Period 5: 0.6228579913928365 0.6290863118622144
Period 6: 0.6156909426076357 0.6218054346492188
Period 7: 0.6092839660687579 0.6152840060390814
Period 8: 0.6034780636858598 0.6093629584532761
Period 9: 0.598165947397939 0.6039352436818675
Period 10: 0.5932899243824563 0.5989437030177499
Period 11: 0.5887859979525862 0.5943242633620437
Period 12: 0.5846168458865866 0.5900401412391936
Period 13: 0.5807349688889774 0.5860436288192699
Period 14: 0.5771066699827863 0.5823011216758961
Period 15: 0.5737155201842229 0.5787966427855701
Period 16: 0.570521026239598 0.5754890635001646
Period 17: 0.5675260194240582 0.572382273463638
Period 18: 0.5647124365814402 0.5694581986882421
Period 19: 0.5620510123777308 0.5666867496108104
Period 20: 0.5595265737735989 0.5640529594891495
Period 21: 0.5571498042189259 0.5615684315566639
Period 22: 0.5549018259035892 0.5592140287749348
Period 23: 0.5527639287462933 0.5569706065637489
Period 24: 0.5507376578961869 0.5548400067629832
Period 25: 0.5487996021972823 0.5527984042975036
Period 26: 0.5469479830767151 0.5508437819761439
Period 27: 0.5451965438547228 0.5489913592678084
Period 28: 0.5435331293043427 0.5472284531451985
Period 29: 0.541948537160021 0.5455457116538099

********** Training Finished With **********

learning rate 5e-07
steps 500000
batch_size 10

********** Weights **********


global_step 500010
linear/linear_model/age/weights [[0.0005494]]
linear/linear_model/age/weights/part_0/Ftrl [[6.718108e+10]]
linear/linear_model/age/weights/part_0/Ftrl_1 [[-2.8480118e+08]]
linear/linear_model/bias_weights [0.00055954]
linear/linear_model/bias_weights/part_0/Ftrl [24626112.]
linear/linear_model/bias_weights/part_0/Ftrl_1 [-5553443.5]
linear/linear_model/ca/weights [[9.981024e-05]]
linear/linear_model/ca/weights/part_0/Ftrl [[5741845.5]]
linear/linear_model/ca/weights/part_0/Ftrl_1 [[-478333.56]]
linear/linear_model/chol/weights [[0.00055665]]
linear/linear_model/chol/weights/part_0/Ftrl [[1.3346874e+12]]
linear/linear_model/chol/weights/part_0/Ftrl_1 [[-1.2861844e+09]]
linear/linear_model/cp/weights [[0.00056528]]
linear/linear_model/cp/weights/part_0/Ftrl [[68083460.]]
linear/linear_model/cp/weights/part_0/Ftrl_1 [[-9328477.]]
linear/linear_model/exang/weights [[-9.899928e-05]]
linear/linear_model/exang/weights/part_0/Ftrl [[921382.3]]
linear/linear_model/exang/weights/part_0/Ftrl_1 [[190056.16]]
linear/linear_model/fbs/weights [[0.00034351]]
linear/linear_model/fbs/weights/part_0/Ftrl [[1407571.1]]
linear/linear_model/fbs/weights/part_0/Ftrl_1 [[-815094.9]]
linear/linear_model/oldpeak/weights [[0.00028214]]
linear/linear_model/oldpeak/weights/part_0/Ftrl [[10813546.]]
linear/linear_model/oldpeak/weights/part_0/Ftrl_1 [[-1855589.8]]
linear/linear_model/restecg/weights [[0.00055331]]
linear/linear_model/restecg/weights/part_0/Ftrl [[9823649.]]
linear/linear_model/restecg/weights/part_0/Ftrl_1 [[-3468432.5]]
linear/linear_model/sex/weights [[0.00047899]]
linear/linear_model/sex/weights/part_0/Ftrl [[8775053.]]
linear/linear_model/sex/weights/part_0/Ftrl_1 [[-2837810.8]]
linear/linear_model/slope/weights [[0.00057025]]
linear/linear_model/slope/weights/part_0/Ftrl [[66868464.]]
linear/linear_model/slope/weights/part_0/Ftrl_1 [[-9326202.]]
linear/linear_model/thal/weights [[0.00053834]]
linear/linear_model/thal/weights/part_0/Ftrl [[1.07644856e+08]]
linear/linear_model/thal/weights/part_0/Ftrl_1 [[-11170724.]]
linear/linear_model/thalach/weights [[0.00056917]]
linear/linear_model/thalach/weights/part_0/Ftrl [[6.533299e+11]]
linear/linear_model/thalach/weights/part_0/Ftrl_1 [[-9.201104e+08]]
linear/linear_model/trestbps/weights [[0.00055516]]
linear/linear_model/trestbps/weights/part_0/Ftrl [[4.1273852e+11]]
linear/linear_model/trestbps/weights/part_0/Ftrl_1 [[-7.1332506e+08]]
```

![RMSE over Periods](/Figure_1.png?raw=true "RMSE over Periods")
