# only memes, mlp, freezegpt
python3 src/clip_model/train.py --train_data data/memes/train.csv --val_data data/memes/val.csv --images_path data/memes/images --num_layers 4 --out_dir models/.checkpoints --epochs 10 --bs 10 --only_prefix --normalize_prefix --device cpu > output/test_clip.log
python3 src/clip_model/evaluate.py --test_data data/memes/test.csv  --images_path data/memes/images  --bs 20 --only_prefix --normalize_prefix --device cpu > output/clip_memes_mlponly_val10ep.log

# translated + memes, mlp+gpt finetuning
#python3 src/clip_model/train.py --train_data data/memes/train.csv --val_data data/memes/val.csv --images_path data/memes/images --num_layers 4 --out_dir models/.checkpoints --extra-data-epochs 10 --prefix full_mlp_gpt --epochs 5 --bs 15 --normalize_prefix --extra-data --device cpu > output/full_data_mlp_gpt.log
#python3 src/clip_model/evaluate.py --test_data data/memes/test.csv  --images_path data/memes/images  --bs 20 --only_prefix --normalize_prefix --device cpu > output/clip_memes_mlponly_val10ep.log
