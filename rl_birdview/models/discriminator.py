# modified from https://github.com/rohitrango/BC-regularized-GAIL/blob/master/a2c_ppo_acktr/algo/gail.py

from pathlib import Path
from typing import Union, Dict, Tuple, Any
from functools import partial
import time
import gym
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import transforms
from rl_birdview.models.torch_layers import DiscXtMaCNN, RGBDiscXtMaCNN
from PIL import Image, ImageDraw
from stable_baselines3.common.running_mean_std import RunningMeanStd
from tqdm import tqdm


class Discriminator(nn.Module):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        batch_size,
        disc_head_arch=[256, 256],
        rgb_gail=False,
        traj_plot=True,
        start_update=0,
        **kwargs,
    ):

        super(Discriminator, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.rgb_gail = rgb_gail
        self.traj_plot = traj_plot

        self.algo_type = "hgail"
        if "algo_type" in kwargs:
            self.algo_type = kwargs["algo_type"]

        if th.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.optimizer_class = th.optim.Adam

        if self.rgb_gail:
            self.features_extractor = RGBDiscXtMaCNN(
                observation_space, action_space, traj_plot=self.traj_plot
            )
            self.optimizer_kwargs = {"lr": 2.5e-4, "eps": 1e-5}
            self.thre = 10
            self.pre_epoch = 2
            self.epoch = 2

        else:
            self.features_extractor = DiscXtMaCNN(observation_space, action_space)
            self.optimizer_kwargs = {"lr": 2.5e-4, "eps": 1e-5}
            self.thre = 10
            self.pre_epoch = 2
            self.epoch = 2

        # best_so_far
        # self.net_arch = [dict(pi=[256, 128, 64], vf=[128, 64])]
        self.disc_head_arch = list(disc_head_arch)
        self.activation_fn = nn.ReLU

        # TODO: potential feature extractor
        # TODO: potential head

        self._build()
        if self.algo_type == "hgail":
            expert_dataset = ExpertDataset(
                "gail_experts",
                n_routes=8,
                n_eps=1,
            )
        elif self.algo_type == "airl":
            expert_dataset = ExpertTransitionDataset(
                "gail_experts",
                n_routes=8,
                n_eps=1,
            )
        else:
            raise ValueError(f"Unknown algorithm type: {self.algo_type}")
        self.expert_loader = th.utils.data.DataLoader(
            expert_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.max_grad_norm = 0.5
        self.i_update = start_update
        self.cliprew_down = -10
        self.cliprew_up = 10

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        self.disc_debug = {}

    def _build(self) -> None:
        last_layer_dim_disc = self.features_extractor.features_dim
        disc_net = []
        for layer_size in self.disc_head_arch:
            disc_net.append(nn.Linear(last_layer_dim_disc, layer_size))
            disc_net.append(self.activation_fn())
            last_layer_dim_disc = layer_size

        disc_net.append(nn.Linear(last_layer_dim_disc, 1))
        self.disc_head = nn.Sequential(*disc_net).to(self.device)

        # TODO: potential head

        self.optimizer = self.optimizer_class(
            self.parameters(), **self.optimizer_kwargs
        )

    def forward(self, obs_dict, action, next_obs_dict=None):
        """
        used in collect_rollouts(), do not clamp actions
        """
        obs_dict = dict(
            [
                (obs_key, obs_item.to(self.device))
                for obs_key, obs_item in obs_dict.items()
            ]
        )

        if self.rgb_gail:
            rgb_array = []
            for rgb_key in ["central_rgb", "left_rgb", "right_rgb"]:
                rgb_img = obs_dict[rgb_key].float() / 255.0
                rgb_array.append(rgb_img)

            if self.traj_plot:
                traj_plot = obs_dict["traj_plot_rgb"].float() / 255.0
                rgb_array.append(traj_plot)
            rgb = th.cat(rgb_array, dim=1).to(self.device)
            cmd = obs_dict["cmd"].to(self.device)
            traj = obs_dict["traj"].to(self.device)
            state = obs_dict["state"].to(self.device)
            action = action.to(self.device)
            features = self.features_extractor(rgb, cmd, traj, state, action)
            disc = self.disc_head(features)
        else:
            if self.algo_type == "hgail":
                birdview = obs_dict["birdview"].to(self.device)
                birdview = birdview.float() / 255.0
                state = obs_dict["state"].to(self.device)
                action = action.to(self.device)
                features = self.features_extractor(birdview, state, action)
                disc = self.disc_head(features)
            elif self.algo_type == "airl":
                # base reward
                birdview = obs_dict["birdview"].to(self.device)
                birdview = birdview.float() / 255.0
                state = obs_dict["state"].to(self.device)
                action = action.to(self.device)
                features = self.features_extractor(birdview, state, action)
                disc = self.disc_head(features)
                # TODO: shape reward 1. potential feature extractor 2. potential head 3. r = r_b + lambda * r_s(o') - r_s(o)

        return disc

    def get_init_kwargs(self) -> Dict[str, Any]:
        init_kwargs = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            batch_size=self.batch_size,
            disc_head_arch=self.disc_head_arch,
            rgb_gail=self.rgb_gail,
            traj_plot=self.traj_plot,
            start_update=self.i_update,
        )
        return init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        model = cls(**saved_variables["policy_init_kwargs"])
        # Load weights
        model.load_state_dict(saved_variables["policy_state_dict"])
        model.to(device)
        return model, saved_variables["train_init_kwargs"]

    def compute_grad_pen(
        self, expert_obs_dict, expert_action, policy_obs_dict, policy_action, lambda_=10
    ):

        # Change state values
        alpha = th.rand(expert_obs_dict["birdview"].size(0), 1, 1, 1)
        mixup_obs_dict = {}
        expert_obs_dict = dict(
            [
                (obs_key, obs_item.to(self.device))
                for obs_key, obs_item in expert_obs_dict.items()
            ]
        )
        expert_action = expert_action.to(self.device)
        policy_obs_dict = dict(
            [
                (obs_key, obs_item.to(self.device))
                for obs_key, obs_item in policy_obs_dict.items()
            ]
        )
        policy_action = policy_action.to(self.device)
        if self.rgb_gail:
            for obs_key in ["central_rgb", "left_rgb", "right_rgb", "traj_plot_rgb"]:
                alpha_obs = alpha.expand_as(expert_obs_dict[obs_key]).to(
                    expert_obs_dict[obs_key].device
                )
                mixup_obs_dict[obs_key] = (
                    alpha_obs * expert_obs_dict[obs_key]
                    + (1 - alpha_obs) * policy_obs_dict[obs_key]
                )
                mixup_obs_dict[obs_key].requires_grad = True

            alpha = alpha.view(expert_obs_dict["central_rgb"].size(0), 1)

            for obs_key in ["cmd", "traj", "state"]:
                alpha_obs = alpha.expand_as(expert_obs_dict[obs_key]).to(
                    expert_obs_dict[obs_key].device
                )
                mixup_obs_dict[obs_key] = (
                    alpha_obs * expert_obs_dict[obs_key]
                    + (1 - alpha_obs) * policy_obs_dict[obs_key]
                )
                mixup_obs_dict[obs_key].requires_grad = True

        else:
            alpha_obs = alpha.expand_as(expert_obs_dict["birdview"]).to(
                expert_obs_dict["birdview"].device
            )
            mixup_obs_dict["birdview"] = (
                alpha_obs * expert_obs_dict["birdview"]
                + (1 - alpha_obs) * policy_obs_dict["birdview"]
            )
            mixup_obs_dict["birdview"].requires_grad = True

            alpha = alpha.view(expert_obs_dict["birdview"].size(0), 1)

            alpha_obs = alpha.expand_as(expert_obs_dict["state"]).to(
                expert_obs_dict["state"].device
            )
            mixup_obs_dict["state"] = (
                alpha_obs * expert_obs_dict["state"]
                + (1 - alpha_obs) * policy_obs_dict["state"]
            )
            mixup_obs_dict["state"].requires_grad = True

        alpha_action = alpha.expand_as(expert_action).to(expert_action.device)
        mixup_action = alpha_action * expert_action + (1 - alpha_action) * policy_action
        mixup_action.requires_grad = True

        inputs_grad = (*mixup_obs_dict.values(), mixup_action)

        disc = self.forward(mixup_obs_dict, mixup_action)
        ones = th.ones(disc.size()).to(disc.device)

        grad = th.autograd.grad(
            outputs=disc,
            inputs=inputs_grad,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        grad = grad.view(grad.size(0), -1)

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, rollout_buffer):
        loss = 0
        expert_ac_loss = 0
        policy_ac_loss = 0
        g_loss = 0.0
        gp = 0.0
        update_samples = 0
        policy_reward = 0
        expert_reward = 0

        disc_epoch = self.epoch
        if self.i_update < self.thre:
            disc_epoch += (
                (self.pre_epoch - self.epoch) * (self.thre - self.i_update) / self.thre
            )  # Warm up
            disc_epoch = int(disc_epoch)

        with tqdm(total=disc_epoch) as pbar:
            for _ in range(disc_epoch):
                for expert_batch, policy_batch in zip(
                    self.expert_loader, rollout_buffer.get(self.batch_size)
                ):
                    policy_obs_dict = policy_batch.observations
                    policy_action = policy_batch.actions
                    policy_next_obs_dict = None
                    if self.algo_type == "airl":
                        policy_next_obs_dict = policy_batch.next_observations

                    # expert data
                    if self.algo_type == "hgail":
                        expert_obs_dict, expert_action = expert_batch
                    elif self.algo_type == "airl":
                        expert_obs_dict, expert_action, expert_next_obs_dict = (
                            expert_batch
                        )
                        # TEST expert obs dict when not shuffle
                        # print(
                        #     "obs match ",
                        #     verify_next_observation(
                        #         expert_obs_dict, expert_next_obs_dict
                        #     ),
                        # )

                    if (
                        policy_obs_dict["birdview"].size()
                        != expert_obs_dict["birdview"].size()
                    ):
                        print(
                            "!!!!!size of policy obs {} != size of expert obs {}".format(
                                policy_obs_dict["birdview"].size(),
                                expert_obs_dict["birdview"].size(),
                            )
                        )
                        continue

                    policy_d = self.forward(policy_obs_dict, policy_action)
                    policy_reward += policy_d.sum().item()

                    expert_d = self.forward(expert_obs_dict, expert_action)
                    expert_reward += expert_d.sum().item()
                    n_samples = policy_obs_dict["state"].shape[0]

                    # Wasserstein discriminator loss
                    # expert_loss = th.mean(th.tanh(expert_d))
                    # policy_loss = th.mean(th.tanh(policy_d))

                    # expert_ac_loss += (expert_loss).item() * n_samples
                    # policy_ac_loss += (policy_loss).item() * n_samples

                    # wd = expert_loss - policy_loss
                    # grad_pen = self.compute_grad_pen(
                    #     expert_obs_dict, expert_action, policy_obs_dict, policy_action
                    # )

                    # loss += (-wd + grad_pen).item() * n_samples
                    # g_loss += (wd).item() * n_samples
                    # gp += (grad_pen).item() * n_samples

                    # FIXME: binary cross entropy discriminator loss
                    # Binary labels
                    real_labels = th.ones_like(expert_d)
                    fake_labels = th.zeros_like(policy_d)
                    labels = th.cat([real_labels, fake_labels])
                    rewards = th.cat([expert_d, policy_d])
                    bce_logits = nn.BCEWithLogitsLoss()
                    bce_loss = bce_logits(rewards, labels)

                    update_samples += n_samples

                    self.optimizer.zero_grad()

                    # Wasserstein discriminator loss
                    # (-wd + grad_pen).backward()

                    # FIXME: binary cross entropy discriminator loss
                    bce_loss.backward()

                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                pbar.update(1)

        self.disc_debug = {
            "disc/loss": loss / update_samples,
            "disc/policy_reward": policy_reward / update_samples,
            "disc/expert_reward": expert_reward / update_samples,
            "disc/gail_loss": g_loss / update_samples,
            "disc/gradient_penalty": gp / update_samples,
            "disc/expert_loss": expert_ac_loss / update_samples,
            "disc/policy_loss": policy_ac_loss / update_samples,
            "disc/epochs": disc_epoch,
            "disc/update_samples": update_samples,
        }

        self.i_update += 1
        return

    def predict_reward(self, obs_dict, action, gamma, dones, update_rms=True):
        with th.no_grad():
            d = self.forward(obs_dict, action)

            s = th.sigmoid(d)
            reward = -(1 - s).log()
            reward = reward.cpu()
            reward = th.clamp(reward, self.cliprew_down, self.cliprew_up)

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                non_terminal = 1.0 - dones
                self.returns = self.returns * non_terminal * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            ret = reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

            # print(
            #     "d {} reward {} var {} norm rew {}".format(
            #         d.mean().item(), reward.mean().item(), self.ret_rms.var[0], ret
            #     )
            # )
            return ret


def verify_next_observation(obs_dict, next_obs_dict):
    """
    Verify if next_obs_dict is equivalent to the next state of obs_dict

    Args:
        obs_dict (dict): Current observation dictionary with 'birdview' and 'state' keys
        next_obs_dict (dict): Next observation dictionary with 'birdview' and 'state' keys

    Returns:
        bool: True if observations match, False otherwise
    """
    # Check if both dictionaries have the same keys
    if set(obs_dict.keys()) != set(next_obs_dict.keys()):
        print(
            "key not match obs {}, next obs {} ".format(
                obs_dict.keys(), next_obs_dict.keys()
            )
        )
        return False

    # Check birdview arrays
    birdview_match = np.array_equal(
        obs_dict["birdview"][1:],  # All except first item
        next_obs_dict["birdview"][:-1],  # All except last item
    )

    # Check state arrays
    state_match = np.array_equal(
        obs_dict["state"][1:],  # All except first item
        next_obs_dict["state"][:-1],  # All except last item
    )
    return birdview_match and state_match


def traj_plotter(traj, img_path=None):
    img_size = 192
    radius = 10
    color = (255, 255, 255)
    scale = 500
    point_idx = -1
    img = Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    while (point_idx + 1) * 2 <= len(traj):
        if point_idx < 0:
            x = traj[1] * scale
            y = -1 * traj[0] * scale
        elif point_idx == 0:
            x = 0
            y = 0
        else:
            x = traj[point_idx * 2 + 1] * scale
            y = -1 * traj[point_idx * 2] * scale
        x += img_size / 2
        y += img_size - 40
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        point_idx += 1
    image_array = np.zeros((img_size, img_size, 1), dtype=np.uint8)
    image_array[:, :, 0] = np.array(img)[:, :, 0]
    image_array = np.transpose(image_array, [2, 0, 1])
    image_tensor = th.as_tensor(image_array)
    return image_tensor


def traj_plotter_rgb(traj, img_path=None):
    img_size = 192
    radius = 10
    color = (255, 255, 255)
    scale = 500
    point_idx = -1
    img = Image.fromarray(np.zeros((144, 256, 3), dtype=np.uint8))
    draw = ImageDraw.Draw(img)
    while (point_idx + 1) * 2 <= len(traj):
        if point_idx < 0:
            x = traj[1] * scale
            y = -1 * traj[0] * scale
        elif point_idx == 0:
            x = 0
            y = 0
        else:
            x = traj[point_idx * 2 + 1] * scale
            y = -1 * traj[point_idx * 2] * scale
        x += img_size / 2
        y += img_size - 40
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        point_idx += 1
    image_array = np.zeros((144, 256, 1), dtype=np.uint8)
    image_array[:, :, 0] = np.array(img)[:, :, 0]
    image_array = np.transpose(image_array, [2, 0, 1])
    image_tensor = th.as_tensor(image_array)
    return image_tensor


class ExpertDataset(th.utils.data.Dataset):
    def __init__(
        self, dataset_directory, n_routes=1, n_eps=1, route_start=0, ep_start=0
    ):
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []

        for route_idx in range(route_start, route_start + n_routes):
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = (
                    self.dataset_path
                    / ("route_%02d" % route_idx)
                    / ("ep_%02d" % ep_idx)
                )
                route_df = pd.read_json(route_path / "episode.json")
                traj_length = route_df.shape[0]
                self.length += traj_length
                for step_idx in range(traj_length):
                    self.get_idx.append((route_idx, ep_idx, step_idx))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(
                        th.Tensor(route_df.iloc[step_idx]["actions"])
                    )

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]

    def __len__(self):
        return self.length

    def process_image(self, image_path):
        image_array = Image.open(image_path)
        image_array = np.transpose(image_array, [2, 0, 1])
        image_tensor = th.as_tensor(image_array.copy())
        return image_tensor

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx = self.get_idx[j]
        if self.actual_obs[j] is None:
            # Load only the first time, images in uint8 are supposed to be light
            ep_dir = self.dataset_path / "route_{:0>2d}/ep_{:0>2d}".format(
                route_idx, ep_idx
            )
            masks_list = []
            for mask_index in range(1):
                mask_tensor = self.process_image(
                    ep_dir
                    / "birdview_masks/{:0>4d}_{:0>2d}.png".format(step_idx, mask_index)
                )
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            central_rgb = self.process_image(
                ep_dir / "central_rgb/{:0>4d}.png".format(step_idx)
            )
            left_rgb = self.process_image(
                ep_dir / "left_rgb/{:0>4d}.png".format(step_idx)
            )
            right_rgb = self.process_image(
                ep_dir / "right_rgb/{:0>4d}.png".format(step_idx)
            )

            obs_dict = {
                "birdview": birdview,
                "central_rgb": central_rgb,
                "left_rgb": left_rgb,
                "right_rgb": right_rgb,
                "item_idx": j,
            }

            state_dict = self.trajs_states[j]
            for state_key in state_dict:
                obs_dict[state_key] = th.Tensor(state_dict[state_key])

            obs_dict["traj_plot"] = traj_plotter(state_dict["traj"])
            obs_dict["traj_plot_rgb"] = traj_plotter_rgb(state_dict["traj"])
            self.actual_obs[j] = obs_dict
        else:
            obs_dict = self.actual_obs[j]

        return obs_dict, self.trajs_actions[j]


class ExpertTransitionDataset(ExpertDataset):
    def __init__(
        self, dataset_directory, n_routes=1, n_eps=1, route_start=0, ep_start=0
    ):
        self.dataset_path = Path(dataset_directory)
        self.length = 0
        self.get_idx = []
        self.trajs_states = []
        self.trajs_actions = []
        self.trajs_last_states = {}

        for route_idx in range(route_start, route_start + n_routes):
            for ep_idx in range(ep_start, ep_start + n_eps):
                route_path = (
                    self.dataset_path
                    / ("route_%02d" % route_idx)
                    / ("ep_%02d" % ep_idx)
                )
                route_df = pd.read_json(route_path / "episode.json")
                traj_length = route_df.shape[0]
                self.length += traj_length - 1
                for step_idx in range(traj_length - 1):
                    self.get_idx.append((route_idx, ep_idx, step_idx, traj_length - 2))
                    state_dict = {}
                    for state_key in route_df.columns:
                        state_dict[state_key] = route_df.iloc[step_idx][state_key]
                    self.trajs_states.append(state_dict)
                    self.trajs_actions.append(
                        th.Tensor(route_df.iloc[step_idx]["actions"])
                    )
                last_state_dict = {}
                for state_key in route_df.columns:
                    last_state_dict[state_key] = route_df.iloc[traj_length - 1][
                        state_key
                    ]
                self.trajs_last_states[(route_idx, ep_idx)] = last_state_dict

        self.trajs_actions = th.stack(self.trajs_actions)
        self.actual_obs = [None for _ in range(self.length)]
        self.next_obs = [None for _ in range(self.length)]

    def __getitem__(self, j):
        route_idx, ep_idx, step_idx, last_idx = self.get_idx[j]
        if self.actual_obs[j] is None:
            # Load only the first time, images in uint8 are supposed to be light
            ep_dir = self.dataset_path / "route_{:0>2d}/ep_{:0>2d}".format(
                route_idx, ep_idx
            )

            # obs
            masks_list = []
            for mask_index in range(1):
                mask_tensor = self.process_image(
                    ep_dir
                    / "birdview_masks/{:0>4d}_{:0>2d}.png".format(step_idx, mask_index)
                )
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            central_rgb = self.process_image(
                ep_dir / "central_rgb/{:0>4d}.png".format(step_idx)
            )
            left_rgb = self.process_image(
                ep_dir / "left_rgb/{:0>4d}.png".format(step_idx)
            )
            right_rgb = self.process_image(
                ep_dir / "right_rgb/{:0>4d}.png".format(step_idx)
            )

            obs_dict = {
                "birdview": birdview,
                "central_rgb": central_rgb,
                "left_rgb": left_rgb,
                "right_rgb": right_rgb,
                "item_idx": j,
            }

            state_dict = self.trajs_states[j]
            for state_key in state_dict:
                obs_dict[state_key] = th.Tensor(state_dict[state_key])

            obs_dict["traj_plot"] = traj_plotter(state_dict["traj"])
            obs_dict["traj_plot_rgb"] = traj_plotter_rgb(state_dict["traj"])

            # next obs
            masks_list.clear()
            for mask_index in range(1):
                mask_tensor = self.process_image(
                    ep_dir
                    / "birdview_masks/{:0>4d}_{:0>2d}.png".format(
                        step_idx + 1, mask_index
                    )
                )
                masks_list.append(mask_tensor)
            birdview = th.cat(masks_list)

            central_rgb = self.process_image(
                ep_dir / "central_rgb/{:0>4d}.png".format(step_idx + 1)
            )
            left_rgb = self.process_image(
                ep_dir / "left_rgb/{:0>4d}.png".format(step_idx + 1)
            )
            right_rgb = self.process_image(
                ep_dir / "right_rgb/{:0>4d}.png".format(step_idx + 1)
            )

            next_obs_dict = {
                "birdview": birdview,
                "central_rgb": central_rgb,
                "left_rgb": left_rgb,
                "right_rgb": right_rgb,
                "item_idx": j,
            }

            if step_idx < last_idx:
                state_dict = self.trajs_states[j + 1]
            elif step_idx == last_idx:
                state_dict = self.trajs_last_states[(route_idx, ep_idx)]
            else:
                raise Exception(
                    "step idx {} can't be larger than the last idx {}".format(
                        step_idx, last_idx
                    )
                )

            for state_key in state_dict:
                next_obs_dict[state_key] = th.Tensor(state_dict[state_key])

            next_obs_dict["traj_plot"] = traj_plotter(state_dict["traj"])
            next_obs_dict["traj_plot_rgb"] = traj_plotter_rgb(state_dict["traj"])

            self.actual_obs[j] = obs_dict
            self.next_obs[j] = next_obs_dict
        else:
            obs_dict = self.actual_obs[j]
            next_obs_dict = self.next_obs[j]

        return obs_dict, self.trajs_actions[j], next_obs_dict
