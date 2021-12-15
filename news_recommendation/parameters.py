import argparse
from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',
                        type=str,
                        default='NRMS',
                        choices=['NRMS', 'NAML', 'LSTUR', 'TANR'])
    parser.add_argument('--dateset',
                        type=str,
                        default='mind-small',
                        choices=[
                            'mind-small', 'mind-large', 'adressa-1week',
                            'adressa-10weeks'
                        ])
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_batches_show_loss',
                        type=int,
                        default=100,
                        help='Number of batchs to show loss')
    parser.add_argument(
        '--num_batches_validate',
        type=int,
        default=1000,
        help='Number of batchs to check metrics on validation dataset')
    parser.add_argument('--save_checkpoint', type=str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--num_clicked_news_a_user',
                        type=int,
                        default=50,
                        help='Number of sampled click history for each user')
    parser.add_argument('--num_words_title', type=int, default=20)
    parser.add_argument('--num_words_abstract', type=int, default=50)
    parser.add_argument('--word_freq_threshold', type=int, default=1)
    parser.add_argument('--negative_sampling_ratio', type=int, default=2)
    parser.add_argument('--dropout_probability', type=float, default=0.2)
    parser.add_argument('--num_words', type=int, default=70976)
    parser.add_argument('--num_categories', type=int, default=275)
    parser.add_argument('--num_users', type=int, default=50001)
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
    args = parser.parse_args()

    dataset_attributes = {
        'NRMS': {
            'news': ['title'],
            'record': []
        },
        'NAML': {
            'news': ['category', 'subcategory', 'title', 'abstract'],
            'record': []
        },
        'LSTUR': {
            'news': ['category', 'subcategory', 'title'],
            'record': ['user', 'clicked_news_length']
        },
        'TANR': {
            'news': ['category', 'title'],
            'record': []
        }
    }

    args.dataset_attributes = dataset_attributes[args.model_name]
    return args
