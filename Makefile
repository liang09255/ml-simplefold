eval-model:
	python eval_checkpoint.py artifacts/checkpoints/last.ckpt

train:
	python src/simplefold/train.py experiment=debug_cpu

board:
	tensorboard --logdir artifacts/tensorboard --port 6006