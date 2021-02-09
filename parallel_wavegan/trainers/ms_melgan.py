#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train Parallel WaveGAN."""

import logging
import os

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch

from tensorboardX import SummaryWriter
from tqdm import tqdm
from parallel_wavegan.trainers.general import Trainer, Collater

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class MSMelGANTrainer(Trainer):
    """Customized trainer module for Parallel WaveGAN training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.name_map={'sc': 'spectral_convergence', 'mag': 'log_stft_magnitude', 'mp': 'magnitude_phase', 'wp': 'weighted_phase', 'ph': 'phase', 'mel': 'log_mel'}

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
            if self.config.get('use_2nd_D'):
                state_dict["model"]["discriminator2"] = self.model["discriminator2"].module.state_dict()
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }
            if self.config.get('use_2nd_D'):
                state_dict["model"]["discriminator2"] = self.model["discriminator2"].state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
            #TODO:fix
            if self.config.get('use_2nd_D'):
                self.model["discriminator2"].module.load_state_dict(state_dict["model"]["discriminator2"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
            #TODO:fix
            if self.config.get('use_2nd_D'):
                self.model["discriminator2"].load_state_dict(state_dict["model"]["discriminator2"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        spk_ids, x, y = batch
        spk_ids = spk_ids.to(self.device)
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        #import pdb;pdb.set_trace()
        #from time import time
        #st=time();y_ = self.model["generator"](*x);print(time()-st)

        # reconstruct the signal from multi-band signal
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # multi-resolution sfft loss
        #sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        #self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        #self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        #gen_loss = sc_loss + mag_loss
        res_map = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        gen_loss = 0.0
        for k, l in res_map.items():
            self.total_train_loss[f"train/{self.name_map[k]}_loss"] += l.item()
            gen_loss += l

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            gen_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            #sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            #self.total_train_loss[
            #    "train/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            #self.total_train_loss[
            #    "train/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            #gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
            sub_res_map = self.criterion["sub_stft"](y_mb_, y_mb)
            for k, l in sub_res_map.items():
                self.total_train_loss[f"train/sub_{self.name_map[k]}_loss"] += l.item()
                gen_loss += 0.5 * l

        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_, spk_ids)
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            else:
                # for multi-scale discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss /= (i + 1)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y, spk_ids)
                    fm_loss = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss /= (i + 1) * (j + 1)
                    self.total_train_loss["train/feature_matching_loss"] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss
            if self.config.get('use_2nd_D'):
                p_ = self.model["discriminator2"](y_, spk_ids)
                if not isinstance(p_, list):
                    # for standard discriminator
                    adv_loss2 = self.criterion["mse"](p_, p_.new_ones(p_.size()))
                    self.total_train_loss["train/adversarial_loss2"] += adv_loss2.item()
                else:
                    # for multi-scale discriminator
                    adv_loss2 = 0.0
                    for i in range(len(p_)):
                        adv_loss2 += self.criterion["mse"](
                            p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                    adv_loss2 /= (i + 1)
                    self.total_train_loss["train/adversarial_loss2"] += adv_loss2.item()

                    # feature matching loss
                    if self.config["use_feat_match_loss"]:
                        # no need to track gradients
                        with torch.no_grad():
                            p = self.model["discriminator2"](y, spk_ids)
                        fm_loss2 = 0.0
                        for i in range(len(p_)):
                            for j in range(len(p_[i]) - 1):
                                fm_loss2 += self.criterion["l1"](p_[i][j], p[i][j].detach())
                        fm_loss2 /= (i + 1) * (j + 1)
                        self.total_train_loss["train/feature_matching_loss2"] += fm_loss2.item()
                        adv_loss2 += self.config["lambda_feat_match"] * fm_loss2

            # add adversarial loss to generator loss
            gen_loss += self.config["lambda_adv"] * adv_loss
            if self.config.get('use_2nd_D'):
                gen_loss += self.config["lambda_adv"] * adv_loss2

        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            if self.config.get("regenerate_fake"):
                with torch.no_grad():
                    y_ = self.model["generator"](*x)
                if self.config["generator_params"]["out_channels"] > 1:
                    y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            p = self.model["discriminator"](y, spk_ids)
            p_ = self.model["discriminator"](y_.detach(), spk_ids)
            if not isinstance(p, list):
                # for standard discriminator
                real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss = real_loss + fake_loss
            else:
                # for multi-scale discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss /= (i + 1)
                fake_loss /= (i + 1)
                dis_loss = real_loss + fake_loss

            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # discriminator loss 2
            if self.config.get('use_2nd_D'):
                p = self.model["discriminator2"](y, spk_ids)
                p_ = self.model["discriminator2"](y_.detach(), spk_ids)
                if not isinstance(p, list):
                    # for standard discriminator
                    real_loss2 = self.criterion["mse"](p, p.new_ones(p.size()))
                    fake_loss2 = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                    dis_loss2 = real_loss2 + fake_loss2
                else:
                    # for multi-scale discriminator
                    real_loss2 = 0.0
                    fake_loss2 = 0.0
                    for i in range(len(p)):
                        real_loss2 += self.criterion["mse"](
                            p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                        fake_loss2 += self.criterion["mse"](
                            p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                    real_loss2 /= (i + 1)
                    fake_loss2 /= (i + 1)
                    dis_loss2 = real_loss2 + fake_loss2

                self.total_train_loss["train/real_loss2"] += real_loss2.item()
                self.total_train_loss["train/fake_loss2"] += fake_loss2.item()
                self.total_train_loss["train/discriminator_loss2"] += dis_loss2.item()

                dis_loss += dis_loss2

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        spk_ids, x, y = batch
        spk_ids = spk_ids.to(self.device)
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x)
        if self.config["generator_params"]["out_channels"] > 1:
            y_mb_ = y_
            y_ = self.criterion["pqmf"].synthesis(y_mb_)

        # multi-resolution stft loss
        #sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        #aux_loss = sc_loss + mag_loss
        res_map = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        #aux_loss = 0.0
        gen_loss = 0.0
        for k, l in res_map.items():
            self.total_eval_loss[f"eval/{self.name_map[k]}_loss"] += l.item()
            #aux_loss += l
            gen_loss += l

        # subband multi-resolution stft loss
        if self.config.get("use_subband_stft_loss", False):
            #aux_loss *= 0.5  # for balancing with subband stft loss
            gen_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.criterion["pqmf"].analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2))  # (B, C, T) -> (B x C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2))  # (B, C, T) -> (B x C, T)
            #sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
            #self.total_eval_loss[
            #    "eval/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            #self.total_eval_loss[
            #    "eval/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            #aux_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
            sub_res_map = self.criterion["sub_stft"](y_mb_, y_mb)
            for k, l in sub_res_map.items():
                self.total_eval_loss[f"eval/sub_{self.name_map[k]}_loss"] += l.item()
                #aux_loss += 0.5 * l
                gen_loss += 0.5 * l

        # adversarial loss
        p_ = self.model["discriminator"](y_, spk_ids)
        if not isinstance(p_, list):
            # for standard discriminator
            adv_loss = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            #gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss
        else:
            # for multi-scale discriminator
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
            adv_loss /= (i + 1)
            #gen_loss = aux_loss + self.config["lambda_adv"] * adv_loss

            # feature matching loss
            if self.config["use_feat_match_loss"]:
                p = self.model["discriminator"](y, spk_ids)
                fm_loss = 0.0
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criterion["l1"](p_[i][j], p[i][j])
                fm_loss /= (i + 1) * (j + 1)
                self.total_eval_loss["eval/feature_matching_loss"] += fm_loss.item()
                #gen_loss += self.config["lambda_adv"] * self.config["lambda_feat_match"] * fm_loss
                adv_loss += self.config["lambda_feat_match"] * fm_loss

        if self.config.get('use_2nd_D'):
            p_ = self.model["discriminator2"](y_, spk_ids)
            if not isinstance(p_, list):
                # for standard discriminator
                adv_loss2 = self.criterion["mse"](p_, p_.new_ones(p_.size()))
            else:
                # for multi-scale discriminator
                adv_loss2 = 0.0
                for i in range(len(p_)):
                    adv_loss2 += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_ones(p_[i][-1].size()))
                adv_loss2 /= (i + 1)

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator2"](y, spk_ids)
                    fm_loss2 = 0.0
                    for i in range(len(p_)):
                        for j in range(len(p_[i]) - 1):
                            fm_loss2 += self.criterion["l1"](p_[i][j], p[i][j].detach())
                    fm_loss2 /= (i + 1) * (j + 1)
                    adv_loss2 += self.config["lambda_feat_match"] * fm_loss2

        gen_loss += self.config["lambda_adv"] * adv_loss
        if self.config.get('use_2nd_D'):
            gen_loss += self.config["lambda_adv"] * adv_loss2

        #######################
        #    Discriminator    #
        #######################
        p = self.model["discriminator"](y, spk_ids)
        p_ = self.model["discriminator"](y_, spk_ids)

        # discriminator loss
        if not isinstance(p_, list):
            # for standard discriminator
            real_loss = self.criterion["mse"](p, p.new_ones(p.size()))
            fake_loss = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
            dis_loss = real_loss + fake_loss
        else:
            # for multi-scale discriminator
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](
                    p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                fake_loss += self.criterion["mse"](
                    p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
            real_loss /= (i + 1)
            fake_loss /= (i + 1)
            dis_loss = real_loss + fake_loss

        # discriminator loss 2
        if self.config.get('use_2nd_D'):
            p = self.model["discriminator2"](y, spk_ids)
            p_ = self.model["discriminator2"](y_, spk_ids)
            if not isinstance(p, list):
                # for standard discriminator
                real_loss2 = self.criterion["mse"](p, p.new_ones(p.size()))
                fake_loss2 = self.criterion["mse"](p_, p_.new_zeros(p_.size()))
                dis_loss2 = real_loss2 + fake_loss2
            else:
                # for multi-scale discriminator
                real_loss2 = 0.0
                fake_loss2 = 0.0
                for i in range(len(p)):
                    real_loss2 += self.criterion["mse"](
                        p[i][-1], p[i][-1].new_ones(p[i][-1].size()))
                    fake_loss2 += self.criterion["mse"](
                        p_[i][-1], p_[i][-1].new_zeros(p_[i][-1].size()))
                real_loss2 /= (i + 1)
                fake_loss2 /= (i + 1)
                dis_loss2 = real_loss2 + fake_loss2

        # add to total eval loss
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        #self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        #self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        self.total_eval_loss["eval/real_loss"] += real_loss.item()
        self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
        self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()
        if self.config.get('use_2nd_D'):
            self.total_eval_loss["eval/real_loss2"] += real_loss2.item()
            self.total_eval_loss["eval/fake_loss2"] += fake_loss2.item()
            self.total_eval_loss["eval/discriminator_loss2"] += dis_loss2.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        _, x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)
        if self.config["generator_params"]["out_channels"] > 1:
            y_batch_ = self.criterion["pqmf"].synthesis(y_batch_)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class MSMelGANCollater(Collater):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False,
                 return_utt_id=False,
                 return_spk_id=False,
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.return_utt_id = return_utt_id
        self.return_spk_id = return_spk_id

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Target signal batch (B, 1, T).

        """
        # check length
        batch = [self._adjust_length(*b) for b in batch if len(b[-1]) > self.mel_threshold]
        if not self.return_spk_id and not self.return_utt_id:
            xs, cs = [b[0] for b in batch], [b[1] for b in batch]
        else:
            tags, xs, cs = [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        c_starts = start_frames - self.aux_context_window
        c_ends = start_frames + self.batch_max_frames + self.aux_context_window
        y_batch = [x[start: end] for x, start, end in zip(xs, x_starts, x_ends)]
        c_batch = [c[start: end] for c, start, end in zip(cs, c_starts, c_ends)]

        # convert each batch to tensor, asuume that each item in batch has the same length
        if self.return_spk_id or self.return_utt_id:
            t_batch = torch.tensor(tags, dtype=torch.long) # (B, N)
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            if self.return_spk_id or self.return_utt_id:
                return t_batch, (z_batch, c_batch), y_batch
            else:
                return (z_batch, c_batch), y_batch
        else:
            if self.return_spk_id or self.return_utt_id:
                return t_batch, (c_batch,), y_batch
            else:
                return (c_batch,), y_batch

    def _adjust_length(self, *args):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(args)==2:
            x, c = args
        elif len(args)==3:
            tag, x, c = args
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        if len(args)==2:
            return x, c
        elif len(args)==3:
            return tag, x, c

