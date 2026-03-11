This folder stores archived checkpoints to avoid accidental overwrite.

Latest backup created by assistant:
- checkpoints/20260312-010937/ckpt.pt

How to use this project checkpoint:

1) Sample from current training output (recommended)
./venv/bin/python sample.py --out_dir=out-shakespeare-char --num_samples=3 --max_new_tokens=200 --start="KING:"

2) Resume training from current output
./venv/bin/python -u train.py config/train_shakespeare_char.py --init_from=resume --out_dir=out-shakespeare-char

3) Restore an archived checkpoint back to active output
cp checkpoints/20260312-010937/ckpt.pt out-shakespeare-char/ckpt.pt

4) Start a brand new run without overwriting old one
./venv/bin/python -u train.py config/train_shakespeare_char.py --out_dir=out-shakespeare-char-v2
