# SFDA using Pseudo Labels
To train the model 
```bash
python train_sfda.py --train_file practice_text/train.tsv \
--train_pred ../outputs/negation/train_pred.tsv \
--update_freq 100 \
--output_dir ../outputs/negation/model/ \
--eval_pred practice_text/dev_labels.txt  \
--eval_file practice_text/dev.tsv \
--APM_Strategy "top_k" \
--top_k 100 \
--alpha_routine "sqr" \
--cf_ratio 20 \
--mlm_pretrain \
--mlm_lr 5e-6 
```
**Please Note:** You may need to extract **train_pred.tsv** first. To do so, make use of **Run_Save.ipynb** notebook

update_freq allows the user to choose the number of global steps after which APM prototype updates will take place. The script will automatically create a model directory titled "top-k" where 'k' is the value passed to top_k flag, and store the predicitons in the file itself.
