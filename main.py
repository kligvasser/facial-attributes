import argparse
import torch
import logging
import signal
import sys
import os
import torch.backends.cudnn as cudnn

from datetime import datetime
from os import path
from random import randint

import utils.misc
import models.misc

from data.datasets import BalancedSampling
from trainer import Trainer


# torch.autograd.set_detect_anomaly(True)


def get_arguments():
    parser = argparse.ArgumentParser(description='Facial attributes classification')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument(
        '--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3)'
    )
    parser.add_argument('--model-config', default='', help='additional architecture configuration')
    parser.add_argument(
        '--model-to-load', default=None, type=str, help='additional architecture configuration'
    )
    parser.add_argument('--train-pkl', default='', type=str, required=True, help='')
    parser.add_argument('--test-pkl', default='', type=str, required=True, help='')
    parser.add_argument('--root-images', default='', type=str, required=True, help='')
    parser.add_argument('--max-size', default=None, type=int, help='')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch-size (default: 128)')
    parser.add_argument('--epochs', default=5, type=int, help='')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr (default: 1e-4)')
    parser.add_argument(
        '--betas', default=[0.9, 0.999], nargs=2, type=float, help='adam betas (default: 0.5 0.9)'
    )
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='')
    parser.add_argument('--step-size', default=3, type=int, help='scheduler step size (default: 8)')
    parser.add_argument('--gamma', default=0.1, type=float, help='scheduler gamma (default: 0.5)')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--print-every', default=50, type=int, help='print-every (default: 1)')
    parser.add_argument('--save-every', default=1, type=int, help='print-every (default: 1)')
    parser.add_argument(
        '--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir'
    )
    parser.add_argument('--save-dir', default=None, type=str, help='')
    parser.add_argument('--evaluation', default=False, action='store_true', help='')
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.save_path = path.join(args.results_dir, args.save_dir)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args


def main():
    # arguments
    args = get_arguments()

    torch.manual_seed(args.seed)

    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # set logs
    utils.misc.mkdir(args.save_path)
    utils.misc.mkdir(os.path.join(args.save_path, 'checkpoints'))
    utils.misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)

    # set model
    model = models.misc.load_model(args, args.model_to_load)
    logging.info(model)
    logging.info(
        'Number of parameters in model: {}\n'.format(
            sum([l.nelement() for l in model.parameters()])
        )
    )

    # loaders
    dataset_train = BalancedSampling(args.train_pkl, args.root_images, True, args.max_size)
    dataset_eval = BalancedSampling(args.test_pkl, args.root_images, False, args.max_size)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # trainer
    trainer = Trainer(args, model, loader_train, loader_eval)

    if args.evaluation:
        trainer.eval()
    else:
        trainer.train(args.epochs)

    return


if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()
