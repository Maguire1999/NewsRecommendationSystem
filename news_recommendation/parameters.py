import argparse
import pandas as pd

from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


# TODO recheck all the parameters


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default='NRMS',
                        choices=['NRMS', 'NAML', 'LSTUR', 'TANR', 'FedNRMS'])
    parser.add_argument('--loss',
                        type=str,
                        default='CE',
                        choices=['BCE', 'CE'])
    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--dataset',
                        type=str,
                        default='mind-small',
                        choices=[
                            'mind-small', 'mind-large', 'adressa-1week',
                            'adressa-10weeks'
                        ])

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_epochs_validate', type=int, default=1)
    parser.add_argument('--num_batches_show_loss', type=int, default=100)

    parser.add_argument('--num_rounds', type=int, default=10000)
    parser.add_argument('--num_rounds_validate', type=int, default=1000)
    parser.add_argument('--num_rounds_show_loss', type=int, default=50)
    parser.add_argument('--num_users_per_round', type=int, default=50)

    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_checkpoint', type=str2bool, default=False)
    parser.add_argument('--cache_dataset', type=str2bool, default=True)
    parser.add_argument('--max_training_dataset_cache_num',
                        type=int,
                        default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_history',
                        type=int,
                        default=50,
                        help='Number of sampled click history for each user')
    parser.add_argument('--num_words_title', type=int, default=20)
    parser.add_argument('--num_words_abstract', type=int, default=50)
    parser.add_argument('--word_frequency_threshold', type=int,
                        default=1)  # TODO
    parser.add_argument('--negative_sampling_ratio', type=int, default=4)
    parser.add_argument('--dropout_probability', type=float, default=0.2)
    parser.add_argument('--num_words', type=int, default=None)
    parser.add_argument('--num_categories', type=int, default=None)
    parser.add_argument('--num_users', type=int, default=None)
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--category_embedding_dim', type=int, default=100)
    parser.add_argument('--query_vector_dim', type=int, default=200)
    parser.add_argument('--num_filters', type=int, default=300)
    parser.add_argument('--window_size', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=15)
    parser.add_argument('--long_short_term_method',
                        type=str,
                        default='ini',
                        choices=['ini', 'con'])
    parser.add_argument('--masking_probability', type=float, default=0.5)
    parser.add_argument('--topic_classification_loss_weight',
                        type=float,
                        default=0.1)
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--tensorboard_runs_dir', type=str, default='./runs/')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument('--show_similarity', type=str2bool, default=True)
    args, extra_args = parser.parse_known_args()

    dataset_attributes = {
        'NRMS': {
            'news': ['title'],
            'behaviors':
            ['history', 'positive_candidates', 'negative_candidates']
        },
        'NAML': {
            'news': ['category', 'subcategory', 'title', 'abstract'],
            'behaviors':
            ['history', 'positive_candidates', 'negative_candidates']
        },
        'LSTUR': {
            'news': ['category', 'subcategory', 'title'],
            'behaviors': [
                'user', 'history', 'history_length', 'positive_candidates',
                'negative_candidates'
            ]
        },
        'TANR': {
            'news': ['category', 'title'],
            'behaviors':
            ['history', 'positive_candidates', 'negative_candidates']
        },
        'FedNRMS': {
            'news': ['title'],
            'behaviors':
            ['user', 'history', 'positive_candidates', 'negative_candidates']
        },
    }
    args.dataset_attributes = dataset_attributes[args.model]

    try:
        if args.num_words is None:
            args.num_words = len(
                pd.read_table(f'data/{args.dataset}/word2int.tsv')) + 1
        if args.num_categories is None:
            args.num_categories = len(
                pd.read_table(f'data/{args.dataset}/category2int.tsv')) + 1
        if args.num_users is None:
            args.num_users = len(
                pd.read_table(f'data/{args.dataset}/user2int.tsv')) + 1
    except FileNotFoundError:
        # Suppress the error if is running data processing
        pass

    return args, extra_args
