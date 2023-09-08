import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    hamming_loss,
)

import models.misc
import utils.recorder
import utils.misc


class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = args.device

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.losses = {
            'train': defaultdict(list),
            'eval': defaultdict(list),
        }

        self.writer = utils.recorder.RecoderX(args.save_path)

        self.train_steps = len(train_loader.dataset) // args.batch_size
        self.eval_steps = len(val_loader.dataset) // args.batch_size

        logging.info('Training steps in epoch: {}.'.format(self.train_steps))
        logging.info('Evaluating steps in epoch: {}.'.format(self.eval_steps))

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch=epoch)
            self.eval_epoch(epoch=epoch)

            if epoch % self.args.save_every == 0:
                models.misc.save_model(
                    self.model,
                    os.path.join(
                        self.args.save_path, 'checkpoints', 'classifier_check_{}.pt'.format(epoch)
                    ),
                )
            logging.info(
                'Epoch: {}, Train loss: {:.4f}, Val loss {:.4f}, Val accuracy {:.2f}, Val hamming {:.2f}, Val average-precision {:.2f}'.format(
                    epoch + 1,
                    np.mean(self.losses['train']['loss'][-self.train_steps :]),
                    np.mean(self.losses['eval']['loss'][-self.eval_steps :]),
                    self.losses['eval']['acc'][-1],
                    self.losses['eval']['hmm'][-1],
                    self.losses['eval']['auprc'][-1],
                )
            )

            self.writer.add_scalar(
                'epoch/loss/train',
                np.mean(self.losses['train']['loss'][-self.train_steps :]),
                epoch,
            )
            self.writer.add_scalar(
                'epoch/loss/eval', np.mean(self.losses['eval']['loss'][-self.eval_steps :]), epoch
            )
            self.writer.add_scalar(
                'epoch/accuracy/eval',
                np.mean(self.losses['eval']['accuracy'][-self.eval_steps :]),
                epoch,
            )

        models.misc.save_model(
            self.model, os.path.join(self.args.save_path, 'checkpoints', 'classifier_check_last.pt')
        )
        models.misc.save_model_entire(
            self.model,
            os.path.join(self.args.save_path, 'checkpoints', 'classifier_check_last_entire.pt'),
        )
        self.writer.close()

    def eval(self):
        self.eval_epoch(epoch=0)

        logging.info(
            'Evaluation: Val loss {:.4f}, Val accuracy {:.2f}, Val average-precision {:.2f}'.format(
                np.mean(self.losses['eval']['loss'][:]),
                np.mean(self.losses['eval']['accuracy'][:]),
                self.losses['eval']['auprc'][-1],
            )
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.scheduler.step(epoch=epoch)
        for step, data in enumerate(self.train_loader):
            self.train_step(data)
            if step % self.args.print_every == 0:
                logging.info(
                    'Step: {}, Loss: {:.4f}'.format(
                        step,
                        self.losses['train']['loss'][-1],
                    )
                )

    def eval_epoch(self, epoch):
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for _, data in enumerate(self.val_loader):
                labels, probabilities, predictions = self.eval_step(data)
                all_labels.append(labels)
                all_probabilities.append(probabilities)
                all_predictions.append(predictions)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probabilities = np.concatenate(all_probabilities, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        thresholds = self._compute_optimal_thresholds(all_labels, all_probabilities)
        thresholds.update({'cls': self.val_loader.dataset.cls_list})
        utils.misc.save_dict(
            thresholds,
            os.path.join(self.args.save_path, 'checkpoints', 'thresholds_{}.pkl'.format(epoch)),
        )

        auprc = average_precision_score(all_labels, all_probabilities)
        self.writer.add_scalar('average_precision/eval', auprc, epoch)
        self.losses['eval']['auprc'].append(auprc)

        acc = accuracy_score(all_labels, all_predictions)
        self.writer.add_scalar('accuracy/eval', acc, epoch)
        self.losses['eval']['acc'].append(acc)

        hmm = hamming_loss(all_labels, all_predictions)
        self.writer.add_scalar('hammin/eval', hmm, epoch)
        self.losses['eval']['hmm'].append(hmm)

        self.writer.plot_multi_confusion_matrices(
            'confusion/eval', all_labels, all_predictions, self.val_loader.dataset.cls_list, epoch
        )
        self.writer.plot_multi_precision_recall_curves(
            'precision_recall/eval',
            all_labels,
            all_probabilities,
            self.val_loader.dataset.cls_list,
            epoch,
        )
        self.writer.plot_multi_roc_curves(
            'roc/eval',
            all_labels,
            all_probabilities,
            self.val_loader.dataset.cls_list,
            epoch,
        )

    def _compute_optimal_thresholds(self, labels, probabilities):
        num_classes = labels.shape[-1]
        optimal_thresholds_pr, optimal_thresholds_roc = np.zeros(num_classes), np.zeros(num_classes)

        for i in range(num_classes):
            precision, recall, thresholds = precision_recall_curve(
                y_true=labels[:, i], probas_pred=probabilities[:, i]
            )
            f1_scores = 2 * precision * recall / (precision + recall)
            optimal_thresholds_pr[i] = thresholds[np.nanargmax(f1_scores)]

            fpr, tpr, thresholds = roc_curve(y_true=labels[:, i], y_score=probabilities[:, i])
            optimal_thresholds_roc[i] = thresholds[np.nanargmax(tpr - fpr)]

        average_thresholds = (optimal_thresholds_roc + optimal_thresholds_pr) / 2.0

        thresholds = {
            'precision_recall': optimal_thresholds_pr,
            'roc': optimal_thresholds_roc,
            'average': average_thresholds,
        }

        return thresholds

    def train_step(self, data):
        inputs = data['input'].to(self.device)
        labels = data['label'].to(self.device)

        self.optimizer.zero_grad()

        preds = self.model(inputs)
        loss = self.criterion(preds, labels)

        loss.backward()
        self.optimizer.step()

        self.losses['train']['loss'].append(loss.item())
        self.writer.add_scalar('loss/train', loss.item(), len(self.losses['train']['loss']))

    def eval_step(self, data):
        inputs = data['input'].to(self.device)
        labels = data['label'].to(self.device)

        probabilities = self.model(inputs)
        loss = self.criterion(probabilities, labels)

        predictions = (torch.sigmoid(probabilities) > 0.5).float()
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

        self.losses['eval']['loss'].append(loss.item())
        self.losses['eval']['accuracy'].append(accuracy)
        self.writer.add_scalar('loss/eval', loss.item(), len(self.losses['eval']['loss']))
        self.writer.add_scalar('accuracy/eval', accuracy, len(self.losses['eval']['accuracy']))

        labels = labels.float().cpu()
        probabilities = probabilities.float().cpu()
        predictions = predictions.float().cpu()

        return labels, probabilities, predictions
