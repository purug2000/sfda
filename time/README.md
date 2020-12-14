# SFDA using Pseudo Labels for Time Expression
To train the model 
```bash
python train_time.py -t practice_text/ -a ../practice_data/ref/time/ -s ../outputs/time -p ../outputs/time_pred
```
For prediction 
```bash
python run_time.py -p practice_text/ -o ../practice_data/res/time -m ../outputs/time
```
