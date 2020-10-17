# SFDA using Pseudo Labels
To train the model 
```bash
python train_sfda.py --train_file practice_text/train.tsv \
--train_pred ../outputs/negation/train_pred.tsv \
--update_freq 100 \
--output_dir ../outputs/negation/model/ \
--eval_pred practice_text/dev_labels.txt  \
--eval_file practice_text/dev.tsv 
```
**Please Note:** You may need to extract **train_pred.tsv** first. To do so, make use of **Run_Save.ipynb** notebook

update_freq allows the user to choose the number of global steps after which APM prototype updates will take place.