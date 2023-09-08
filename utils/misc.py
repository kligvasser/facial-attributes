import pickle
import pandas as pd
import logging.config
import os


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_dict(d, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(d, f)


def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    return loaded_dict


def save_df(df, df_path, index=False):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df.to_csv(df_path, index=index)
    elif ext == 'xlsx':
        df.to_excel(df_path, index=index)
    elif ext == 'json':
        df.to_json(df_path, orient='records')
    elif ext == 'parquet':
        df.to_parquet(df_path, index=index)
    elif ext == 'pkl':
        df.to_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pkl'."
        )


def load_df(df_path):
    ext = df_path.split('.')[-1]

    if ext == 'csv':
        df = pd.read_csv(df_path)
    elif ext == 'xlsx':
        df = pd.read_excel(df_path)
    elif ext == 'json':
        df = pd.read_json(df_path)
    elif ext == 'parquet':
        df = pd.read_parquet(df_path)
    elif ext == 'pkl':
        df = pd.read_pickle(df_path)
    else:
        raise ValueError(
            "Unsupported file format. Please choose from 'csv', 'xlsx', 'json', 'parquet', or 'pkl'."
        )

    return df


def setup_logging(log_file='log.txt', resume=False, dummy=False):
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_file,
            filemode=file_mode,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
