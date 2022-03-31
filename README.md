# TextDefender

Codes for "Searching for an Effective Defender:Benchmarking Defense against Adversarial Word Substitution" (EMNLP2021)

### How to run our codes

if you want to train a model from scratch:

**python** main.py **--mode** train **--dataset_name** agnews **--max_seq_length** 128 **--epochs** 10 **--batch_size** 32 **--training_type** base(or freelb, pgd, etc.)

if you want to attack a trained model:

**python** main.py **--mode** attack **--attack_method** textfooler **--attack_numbers** 1000 **--dataset_name** agnews **--max_seq_length** 128 **--batch_size** 32 **--training_type** base(or freelb, pgd, etc.)
