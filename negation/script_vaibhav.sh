#script to run train_sfda.py
python sfda/negation/train_sfda.py --train_file sfda/negation/practice_text/train.tsv \
--train_pred sfda/negation/../outputs/negation/train_pred.tsv \
--update_freq 100 \
--output_dir sfda/negation/../outputs/negation/model/ \
--eval_pred sfda/negation/dev_ta/dev_labels.txt  \
--eval_file sfda/negation/dev_ta/dev.tsv \
--APM_Strategy "top_k" \
--top_k 100 \
--alpha_routine "sqr" \
--cf_ratio 10 \
--do_mlm \
--mlm_lr 5e-6 
