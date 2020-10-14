#SFDA using missing labels
TO train the model 
```bash
python train_sfda.py \
--train_file practice_test/train.tsv\
--train_pred ../outputs/negation/train_pred.tsv \
--output_dir ../outputs/negation/model/ \ 
```
** Please Note:** You may need to extract **train_pred.tsv** first. To do so, make use of Run_Save.ipynb notebook

