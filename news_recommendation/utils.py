import time
import os
import logging
import coloredlogs
import datetime


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def create_logger(args):
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG',
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')
    log_dir = os.path.join(args.log_path, f'{args.model}-{args.dataset}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir,
        f"{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}.txt"
    )
    logger.info(f'Check {log_file_path} for the log of this run')
    file_hander = logging.FileHandler(log_file_path)
    logger.addHandler(file_hander)
    return logger


def dict2table(d, k_fn=str, v_fn=None, corner_name=''):
    '''
    Convert a nested dict to markdown table
    '''
    if v_fn is None:

        def v_fn(x):
            # Precision for abs(x):
            # [0, 0.1)    6
            # [0.1-1)    5
            # [1-10)    4
            # [10-100)    3
            # [100-1000)    2
            # [1000,oo)    1
            precision = max(1, min(5 - math.ceil(math.log(abs(x), 10)), 6))
            return f'{x:.{precision}f}'

    def parse_header(d, depth=0):
        assert depth in [0, 1], 'Only 1d or 2d dicts allowed'
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([corner_name] * depth + list(map(k_fn, d.keys())))} |"

    def parse_segmentation(d):
        return ' --- '.join(['|'] * parse_header(d).count('|'))

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), parse_segmentation(d), *parse_content(d), '\n']
    return '\n'.join(lines)


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])
