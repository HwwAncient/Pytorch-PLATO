"""
Preprocess script.
"""

import os
import argparse

from plato.args import parse_args
from plato.data.dataset import Dataset
from plato.data.field import BPETextField


def main():
    parser = argparse.ArgumentParser()
    
    BPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    
    args = parse_args(parser)
    
    raw_train_file = os.path.join(args.data_dir, "dial.train")
    raw_valid_file = os.path.join(args.data_dir, "dial.valid")
    raw_test_file = os.path.join(args.data_dir, "dial.test")
    train_file = raw_train_file + f".{args.tokenizer_type}.jsonl"
    valid_file = raw_valid_file + f".{args.tokenizer_type}.jsonl"
    test_file = raw_test_file + f".{args.tokenizer_type}.jsonl"
    
    bpe = BPETextField(args.BPETextField)
    
    BUILD_EXAMPLES_FN = {
        "multi": bpe.build_examples_multi_turn,
        "multi_knowledge": bpe.build_examples_multi_turn_with_knowledge
    }
    build_examples_fn = BUILD_EXAMPLES_FN[args.data_type]
    
    if os.path.exists(raw_valid_file) and not os.path.exists(valid_file):
        valid_examples = build_examples_fn(raw_valid_file, data_type="valid")
        bpe.save_examples(valid_examples, valid_file)
    
    if os.path.exists(raw_test_file) and not os.path.exists(test_file):
        test_examples = build_examples_fn(raw_test_file, data_type="test")
        bpe.save_examples(test_examples, test_file)
    
    if os.path.exists(raw_train_file) and not os.path.exists(train_file):
        train_examples = build_examples_fn(raw_train_file, data_type="train")
        bpe.save_examples(train_examples, train_file)

    return


if __name__ == "__main__":
    main()
