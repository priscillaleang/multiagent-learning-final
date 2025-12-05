.PHONY: install test run-kuhn-nfsp run-kuhn-psro

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

run-kuhn-nfsp:
	python experiments/runners/train_nfsp.py --game kuhn_poker --seed 0 --num_episodes 10000

run-kuhn-psro:
	python experiments/runners/train_psro.py --game kuhn_poker --seed 0 --num_iterations 5
