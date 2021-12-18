import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source',
    type=str,
    required=True,
)
parser.add_argument(
    '--target',
    type=str,
    required=True,
)
args = parser.parse_args()

with open(args.source, 'r') as f:
    source = f.read().strip().split('\n')

with open(args.target, 'w') as f:
    for line in tqdm(source):
        begin, end = line.split('\t')[:-1], line.split('\t')[-1]
        begin = '\t'.join(begin)
        end = ' '.join([f'{x}-1' for x in end.split(' ')[:1]] +
                       [f'{x}-0' for x in end.split(' ')[1:]])
        f.write(f'{begin}\t{end}\n')
