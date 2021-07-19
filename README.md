# DRL for real-time scheduling in wireless networks

ND-initialization.py trains a model from the base ND algorithm.

ND-trained-model.py evaluates the performance of the trained model.

Command line:
>adaeq=0:
>>N=10,M=2:
>>>python trainEnsembleQ.py --seed 0 --n 10 --m 2 --adaeq 0 > seed-0-N10M2.log 2>&1 &

>>>python trainEnsembleQ.py --seed 1 --n 10 --m 2 --adaeq 0 > seed-1-N10M2.log 2>&1 &

>>N=10,M=5:
>>>python trainEnsembleQ.py --seed 0 --n 10 --m 5 --adaeq 0 > seed-0-N10M5.log 2>&1 &

>>>python trainEnsembleQ.py --seed 1 --n 10 --m 5 --adaeq 0 > seed-1-N10M5.log 2>&1 &

>>N=2,M=2:
>>>python trainEnsembleQ.py --seed 0 --n 2 --m 2 --adaeq 0 > seed-0-N2M2.log 2>&1 &

>>>python trainEnsembleQ.py --seed 1 --n 2 --m 2 --adaeq 0 > seed-1-N2M2.log 2>&1 &

>adaeq=1:
>>N=10,M=4:
>>>python trainEnsembleQ.py --seed 0 --n 10 --m 4 --adaeq 1 > seed-0.log 2>&1 &

>>>python trainEnsembleQ.py --seed 1 --n 10 --m 4 --adaeq 1 > seed-1.log 2>&1 &
