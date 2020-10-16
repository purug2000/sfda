# SFDA using Pseudo Labels
To train the model 
```bash
python train_sfda.py \
--train_file practice_test/train.tsv\
--train_pred ../outputs/negation/train_pred.tsv \
--output_dir ../outputs/negation/model/ \ 
--update_freq 150 
```
**Please Note:** You may need to extract **train_pred.tsv** first. To do so, make use of **Run_Save.ipynb** notebook

update_freq allows the user to choose the number of global steps after which APM prototype updates will take place.