"""
Trainer class.
"""

import json
import logging
import os
import sys
import time
import torch

import numpy as np
from tqdm import tqdm

from plato.args import str2bool
from plato.data.data_loader import DataLoader
from plato.metrics.metrics_tracker import MetricsTracker
from plato.metrics.metrics import bleu
from plato.metrics.metrics import distinct


def get_logger(log_path, name="default"):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def evaluate_generation_result(results):
    tgt = [result["tgt"].split(" ") for result in results]
    pred = [result["preds"][np.argmax(result["scores"])]
            if isinstance(result["preds"], list)
            else result["preds"]
            for result in results]
    pred = [p.split(" ") for p in pred]
    metrics = {}
    metrics_tracker = MetricsTracker()

    bleu1, bleu2 = bleu(pred, tgt)
    metrics.update({"bleu_1": bleu1, "bleu_2": bleu2})

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(pred)
    metrics.update({"intra_dist_1": intra_dist1,
                    "intra_dist_2": intra_dist2,
                    "inter_dist_1": inter_dist1,
                    "inter_dist_2": inter_dist2})

    avg_len = sum(map(len, pred)) / len(pred)
    metrics.update({"len": avg_len})

    metrics_tracker.update(metrics, num_samples=1)  # 一次更新所有数据的指标到位，没有累积更新，故num_sample取为1
    return metrics_tracker


class Trainer(object):

    @classmethod
    def add_cmdline_argument(cls, parser):
        """ Add the cmdline arguments of trainer. """
        group = parser.add_argument_group("Trainer")
        group.add_argument("--gpu", type=int, default=-1,
                           help="Whether to use gpu for running, default using cpu.")
        group.add_argument("--use_data_distributed", type=str2bool, default=False,
                           help="Whether to use data distributed for parallel training.")
        group.add_argument("--valid_metric_name", type=str, default="-loss",
                           help="The validation metric determining which checkpoint is the best.")
        group.add_argument("--num_epochs", type=int, default=10,
                           help="Total number of training epochs to perform.")
        group.add_argument("--save_dir", type=str, required=True,
                           help="The output directory where the model will be saved.")
        group.add_argument("--batch_size", type=int, default=8,
                           help="Total batch size for training/evaluation/inference.")
        group.add_argument("--log_steps", type=int, default=100,
                           help="The number of training steps to output current metrics "
                           "on past training dataset.")
        group.add_argument("--valid_steps", type=int, default=2000,
                           help="The number of training steps to perform a evaluation "
                           "on validation datasets.")
        group.add_argument("--save_checkpoint", type=str2bool, default=True,
                           help="Whether to save one checkpoints for each training epoch.")
        group.add_argument("--save_summary", type=str2bool, default=False,
                           help="Whether to save metrics summary for visualDL module.")
        DataLoader.add_cmdline_argument(group)
        return group

    def __init__(self, model, to_tensor, hparams, logger=None, lr_scheduler=None):
        self.model = model
        self.to_tensor = to_tensor

        self.is_decreased_valid_metric = hparams.valid_metric_name[0] == "-"
        self.valid_metric_name = hparams.valid_metric_name[1:]
        self.num_epochs = hparams.num_epochs
        self.save_dir = hparams.save_dir
        self.log_steps = hparams.log_steps
        self.valid_steps = hparams.valid_steps
        self.save_checkpoint = hparams.save_checkpoint
        self.save_summary = hparams.save_summary
        self.lr_scheduler = lr_scheduler

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger = logger or get_logger(os.path.join(self.save_dir, "trainer.log"), "trainer")

        self.batch_metrics_tracker = MetricsTracker()
        self.token_metrics_tracker = MetricsTracker()

        self.best_valid_metric = float("inf" if self.is_decreased_valid_metric else "-inf")
        self.epoch = 0
        self.batch_num = 0

    def train_epoch(self, train_iter, valid_iter, infer_iter=None, infer_parse_dict=None):
        """
        Train an epoch.
        产生整个src+tgt端对应的输出，tgt端生成是完全teacher forcing(并行)
        训练和评估时未用到cache，因为并行生成

        @param train_iter
        @type : DataLoader

        @param valid_iter
        @type : DataLoader

        @param infer_iter
        @type : DataLoader

        @param infer_parse_dict
        @type : dict of function
        """
        self.epoch += 1
        num_batches = len(train_iter)
        self.batch_metrics_tracker.clear()
        self.token_metrics_tracker.clear()
        times = []
        for batch_id, (batch, batch_size) in enumerate(train_iter, 1):
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            batch["epoch"] = self.epoch
            batch["num_steps"] = self.batch_num

            # Do a training iteration
            start_time = time.time()
            metrics = self.model(batch, is_training=True)
            token_num = metrics.pop("token_num", None)
            elapsed = time.time() - start_time
            times.append(elapsed)

            batch_metrics = {k: v for k, v in metrics.items() if "token" not in k}
            token_metrics = {k: v for k, v in metrics.items() if "token" in k}
            self.batch_metrics_tracker.update(batch_metrics, batch_size)
            self.token_metrics_tracker.update(token_metrics, token_num)
            self.batch_num += 1

            if self.log_steps and batch_id % self.log_steps == 0:
                batch_metrics_message = self.batch_metrics_tracker.value()
                token_metrics_message = self.token_metrics_tracker.value()
                message_prefix = f"[Train][{self.epoch}][{batch_id}/{num_batches}]"
                avg_time = f"AVG_Time-{sum(times[-self.log_steps:]) / self.log_steps:.3f}"
                message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message,
                                      avg_time])
                self.logger.info(message)

            if self.valid_steps and valid_iter is not None and batch_id % self.valid_steps == 0:
                self.evaluate(valid_iter)

        if valid_iter is not None:
            self.evaluate(valid_iter)

        if infer_iter is not None and infer_parse_dict is not None:
            self.infer(infer_iter, infer_parse_dict)

        return

    def infer(self, data_iter, parse_dict, num_batches=None):
        """
        Inference interface.
        生成时只逐步生成tgt端对应的序列(不能并行)，无teacher forcing
        src端不产生输出，采用cache利用src端数据(会先产生src端的cache再进行tgt端的生成)

        @param : data_iter
        @type : DataLoader

        @param : parse_dict
        @type : dict of function

        @param : num_batches : the number of batch to infer
        @type : int/None
        """
        self.logger.info("Generation starts ...")
        infer_save_file = os.path.join(self.save_dir, f"infer_{self.epoch}.result.json")

        # Inference
        infer_results = []
        batch_cnt = 0
        begin_time = time.time()
        for batch, batch_size in tqdm(data_iter, total=num_batches):
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))

            result = self.model.infer(inputs=batch)
            batch_result = {}

            def to_list(batch):
                """ Parse list. """
                return batch.tolist()

            # parse
            for k in result:
                if k in parse_dict:
                    parse_fn = parse_dict[k]
                else:
                    parse_fn = to_list
                if result[k] is not None:
                    batch_result[k] = parse_fn(result[k])

            for vs in zip(*batch_result.values()):
                infer_result = {}
                for k, v in zip(batch_result.keys(), vs):
                    infer_result[k] = v
                infer_results.append(infer_result)

            batch_cnt += 1
            if batch_cnt == num_batches:
                break

        self.logger.info(f"Saved inference results to {infer_save_file}")
        with open(infer_save_file, "w") as fp:
            json.dump(infer_results, fp, indent=2)
        infer_metrics_tracker = evaluate_generation_result(infer_results)
        metrics_message = infer_metrics_tracker.summary()
        message_prefix = f"[Infer][{self.epoch}]"
        time_cost = f"TIME-{time.time() - begin_time:.3f}"
        message = "   ".join([message_prefix, metrics_message, time_cost])
        self.logger.info(message)
        return

    def evaluate(self, data_iter, need_save=True):
        """
        Evaluation interface
        和训练时一样，产生整个src+tgt端对应的输出，且tgt端生成也是完全teacher forcing

        @param : data_iter
        @type : DataLoader

        @param : need_save
        @type : bool
        """
        # Evaluation
        begin_time = time.time()
        batch_metrics_tracker = MetricsTracker()
        token_metrics_tracker = MetricsTracker()
        for batch, batch_size in data_iter:
            batch = type(batch)(map(lambda kv: (kv[0], self.to_tensor(kv[1])), batch.items()))
            metrics = self.model(batch, is_training=False)
            token_num = int(metrics.pop("token_num"))
            batch_metrics = {k: v for k, v in metrics.items() if "token" not in k}
            token_metrics = {k: v for k, v in metrics.items() if "token" in k}
            batch_metrics_tracker.update(batch_metrics, batch_size)
            token_metrics_tracker.update(token_metrics, token_num)
        batch_metrics_message = batch_metrics_tracker.summary()
        token_metrics_message = token_metrics_tracker.summary()
        message_prefix = f"[Valid][{self.epoch}]"
        time_cost = f"TIME-{time.time() - begin_time:.3f}"
        message = "   ".join([message_prefix, batch_metrics_message, token_metrics_message, time_cost])
        self.logger.info(message)

        if need_save:
            # Check valid metric
            cur_valid_metric = batch_metrics_tracker.get(self.valid_metric_name)
            if self.is_decreased_valid_metric:
                is_best = cur_valid_metric < self.best_valid_metric
            else:
                is_best = cur_valid_metric > self.best_valid_metric
            if is_best:
                self.best_valid_metric = cur_valid_metric
            self.save(is_best)

        return

    def save(self, is_best=False):
        """ save """
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.model.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()

        # Save checkpoint
        if self.save_checkpoint:
            model_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.model")
            torch.save(self.model.state_dict(), model_file)
            self.logger.info(f"Saved model state to '{model_file}'")

            train_file = os.path.join(self.save_dir, f"state_epoch_{self.epoch}.train")
            torch.save(train_state, train_file)
            self.logger.info(f"Saved train state to '{train_file}'")

        # Save current best model
        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            torch.save(self.model.state_dict(), best_model_file)
            best_train_file = os.path.join(self.save_dir, "best.train")
            torch.save(train_state, best_train_file)
            self.logger.info(
                f"Saved best model state to '{best_model_file}' with new best valid metric "
                f"{self.valid_metric_name.upper()}={self.best_valid_metric:.3f}")

    def load(self):
        """ load """
        if self.model.init_checkpoint is None:
            # train from scratch
            # init_checkpoint: None
            return

        if 'PLATO' in self.model.init_checkpoint:
            # load pre-train model, then train
            # init_checkpoint: 'model/PLATO'
            model_state_dict = torch.load(f'{self.model.init_checkpoint}.pt',
                                          map_location=lambda storage, loc: storage)

            parameters = {name: param for name, param in self.model.named_parameters()}
            for name, param in model_state_dict.items():
                if name in parameters:
                    if param.shape != parameters[name].shape:
                        print(f"part of parameter({name}) random normlize initialize")
                        assert hasattr(param, "numpy")
                        arr = param.numpy()
                        z = np.random.normal(scale=self.model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        if name == 'embedder.token_embedding.weight':
                            z[-param.shape[0]:] = arr
                        else:
                            z[:param.shape[0]] = arr
                        dtype, device = param.dtype, param.device
                        z = torch.tensor(z, dtype=dtype, device=device)
                        model_state_dict[name] = z
            for name in parameters:
                if name not in model_state_dict:
                    if parameters[name].requires_grad:
                        print(f"parameter({name}) random normlize initialize")
                        z = np.random.normal(scale=self.model.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        dtype, device = parameters[name].dtype, parameters[name].device
                        model_state_dict[name] = torch.tensor(z, dtype=dtype, device=device)
                    else:
                        model_state_dict[name] = parameters[name]

            self.model.load_state_dict(model_state_dict)
            self.logger.info(f"Loaded pre-train model from '{self.model.init_checkpoint}'")
            return

        if os.path.isfile(os.path.join(self.save_dir, self.model.init_checkpoint)):
            # load fine-tune model, then train
            # init_checkpoint: 'best.model'/'state_epoch_*.model'
            file_prefix = self.model.init_checkpoint.split('.')[0]
            model_file = f"{self.save_dir}/{file_prefix}.model"
            train_file = f"{self.save_dir}/{file_prefix}.train"

            model_state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(model_state_dict)
            self.logger.info(f"Loaded model state from '{model_file}'")

            train_state_dict = torch.load(train_file, map_location=lambda storage, loc: storage)
            self.epoch = train_state_dict["epoch"]
            self.best_valid_metric = train_state_dict["best_valid_metric"]
            self.batch_num = train_state_dict["batch_num"]
            self.model.optimizer.load_state_dict(train_state_dict["optimizer"])
            if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
                self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
            self.logger.info(
                f"Loaded train state from '{train_file}' with (epoch-{self.epoch} "
                f"best_valid_metric={self.best_valid_metric:.3f})")



