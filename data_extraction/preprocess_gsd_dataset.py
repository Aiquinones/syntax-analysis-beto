'''
We preprocess the GSD-Spanish dataset so it's easy to work with the dependencies

Now, each example of the dataset will be an object with:

Example = {
    'id': str, // same as the one in the original dataset, helps when skipping examples
    'words': str[], // a list of the words that form the sentence
    'heads': int[], // a list of the syntactic heads of each word (0 if ROOT, shift 1)
    'relns': str[], // a list of the syntactic relations between the word and its head 
}

We will skip the examples that include chinese characters,
since BERT considers each character a seperate word but the dataset only gives
a relation and head to tthe chinese word (many characters), leaving us with BERT-words
that are not part of the syntactic tree.  
'''

import json
from tqdm import tqdm
import argparse


def _is_chinese_char(cp):
    """
    Checks whether CP is the codepoint of a CJK character.
    From Huggingface's Source code for transformers.tokenization_bert (https://huggingface.co/transformers/_modules/transformers/tokenization_bert.html).
    """
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def has_chinese_char(ex):
    for word in ex['words']:
        for char in word:
            cp = ord(char)
            if _is_chinese_char(cp):
                return True
    return False


def get_word_data(line, ex):
    '''
    We get the data for each word:
    - word
    - head
    - syntactic dependency (with head)

    ** Some words are compound  words (e.g. del = de el)
    The dataset separates compound words into its compounds.
    Hence, we skip compound words, as their compounds are included later.
    '''
    word_id, word, _, _, _, _, head, reln, _, _ = line.strip().split('\t')

    if '-' in word_id:
        # Found a compound word, we skip
        return

    ex['words'].append(word)
    ex['relns'].append(reln)
    ex['heads'].append(int(head))


def get_examples_from_dataset(dataset_filename, output_filename):
    with open(dataset_filename, 'r') as f:
        filtered_examples = 0
        preprocessed_dataset = []
        ex = None

        for line in tqdm(f.readlines()):

            if line == '\n':
                # If line is empty, either:
                # - we're going from one phrase to another -> example is done, append
                # - there are multiple empty lines together -> skip

                if ex:
                    if not has_chinese_char(ex):
                        preprocessed_dataset.append(ex)
                    else:
                        filtered_examples += 1
                    ex = None

            elif line[0] == "#":
                # It's either
                # - # sent_id
                # - # text
                # The text will come later word by word, so we ignore the # text line

                if line[2:6] != "text":  # then it's sent_id
                    # We're processing a new example, create new instance
                    ex = {
                        'id': line.strip().split(' ')[-1],
                        'words': [],
                        'relns': [],
                        'heads': [],
                    }

            elif line[0].isdigit():
                # We add the data for each word
                get_word_data(line, ex)

    if filtered_examples > 0:
        print(
            f'{filtered_examples} examples were filtered for having chinese characters')

    with open(output_filename, 'w') as f:
        json.dump(preprocessed_dataset, f)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Specify variables for the GSD Dataset preprocessing.')

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("--dataset_dir", required=True,
                        help="The directory for the dataset to be preprocessed. The dataset must have the GSD formatting")
    parser.add_argument("--language", default='es',
                        help="language of the preprocessed dataset.")
    parser.add_argument("--partition", default='all',
                        help='partition(s) to be preprocessed. Could be "dev", "train", "test" or "all"')
    parser.add_argument("--output_dir", required=False,
                        help='The directory where the preprocessed dataset will be stored')

    args = parser.parse_args()

    partitions = [args.partition]
    if args.partition == 'all':
        partitions = ['dev', 'test', 'train']

    for partition in partitions:
        dataset_filename = f'{args.dataset_dir}/{args.language}_gsd-ud-{partition}.conllu'
        output_dir = args.output_dir if args.output_dir is not None else args.dataset_dir
        output_filename = f'{output_dir}/{args.language}_gsd-ud-{partition}.json'

        print(f'Preprocessed {partition}...')

        get_examples_from_dataset(dataset_filename, output_filename)
