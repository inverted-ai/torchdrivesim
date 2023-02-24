"""
Imitation learning of driving behavior using INTERACTION dataset.
Uses a simple neural policy consuming the current bird's eye view of the simulation,
and can be trained either with teacher forcing, which corresponds to behavioral cloning,
or without, where gradients are backpropagated through the simulator.
Note that the INTERACTION dataset is subject to its own license terms and needs to
be downloaded separately.
"""
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import trange

import lanelet2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdrivesim.lanelet2 import road_mesh_from_lanelet_map, lanelet_map_to_lane_mesh
from torchdrivesim.behavior.replay import ReplayWrapper
from torchdrivesim.kinematic import SimpleKinematicModel
from torchdrivesim.mesh import BaseMesh, BirdviewMesh
from torchdrivesim.rendering import renderer_from_config
from torchdrivesim.simulator import TorchDriveConfig, Simulator, HomogeneousWrapper
from torchdrivesim.utils import Resolution


def to_device(items, device):
    items_new = {}
    for key, item in items.items():
        if torch.is_tensor(item):
            items_new[key] = item.to(device)
        elif isinstance(item, BaseMesh):
            items_new[key] = item.to(device)
        else:
            items_new[key] = item
    return items_new


def cycle(iterable):  # method for iterating infinitely through dataset
    while True:
        for x in iterable:
            yield x


# INTERACTION Dataset v1.2
class INTERACTIONDataset(torch.utils.data.Dataset):
    agent_type_names = ['vehicle', 'pedestrian']

    def __init__(self, dataset_path, location_names=None, split='train'):
        self.split = split
        self.location_names = []
        self.road_meshes = {}
        self.lane_meshes = {}
        for name in os.listdir(os.path.join(dataset_path, split)):
            name = name[:-(len(split)+5)]
            if location_names is not None:
                if name in location_names:
                    self.location_names.append(name)
                else:
                    continue
            else:
                self.location_names.append(name)
            projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(0, 0))
            lanelet_map = lanelet2.io.load(os.path.join(dataset_path, 'maps', name + '.osm'), projector)
            road_mesh = BirdviewMesh.set_properties(road_mesh_from_lanelet_map(lanelet_map), category='road')
            lane_mesh = lanelet_map_to_lane_mesh(lanelet_map)
            self.road_meshes[name] = road_mesh
            self.lane_meshes[name] = lane_mesh

        self.idx2segment = []
        self.recording_dfs = []
        for location in self.location_names:
            recording_path = os.path.join(dataset_path, split, f'{location}_{split}.csv')
            recording_df = pd.read_csv(recording_path)
            # Pedestrians don't have psi_rad, length and width defined
            recording_df['psi_rad'] = recording_df['psi_rad'].fillna(0)
            recording_df['length'] = recording_df['length'].fillna(1.5)
            recording_df['width'] = recording_df['width'].fillna(1.5)
            recording_df.loc[recording_df['agent_type'] == 'car', 'agent_type'] = 'vehicle'
            recording_df.loc[recording_df['agent_type'] == 'pedestrian/bicycle', 'agent_type'] = 'pedestrian'
            self.recording_dfs.append(recording_df)
            for case_id in recording_df['case_id'].unique():
                case_df = recording_df.loc[recording_df['case_id'] == case_id]
                for track_id in case_df['track_id'].unique():
                    track_df = case_df[(case_df['track_id'] == track_id)]
                    if track_df['agent_type'].iloc[0] != 'vehicle' or len(track_df) != 40:
                        continue
                    self.idx2segment.append({
                        'location': location,
                        'recording_idx': len(self.recording_dfs)-1,
                        'case_id': case_id,
                        'ego_track_id': track_id
                    })

    def subsample(self, num_segments=50, seed=0):
        rng = np.random.default_rng(seed=seed)
        inds = rng.choice(len(self), num_segments, replace=False)
        self.idx2segment = [segment for i, segment in enumerate(self.idx2segment) if i in inds]
        return self

    def __len__(self):
        return len(self.idx2segment)

    def __getitem__(self, idx):
        segment_info = self.idx2segment[idx]
        df = self.recording_dfs[segment_info['recording_idx']]
        ego_id = segment_info['ego_track_id']
        case_id = segment_info['case_id']
        location = segment_info['location']

        df_segment = df[df['case_id'] == case_id].copy()
        df_segment = df_segment.drop('case_id', axis=1)
        df_segment['agent_role'] = 'others'
        df_segment.loc[df_segment.track_id == ego_id, 'agent_role'] = 'agent'
        df_segment = df_segment.sort_values(['agent_role', 'frame_id', 'track_id'],
                                            ascending=[True, True, True], kind='mergesort')

        agent_ids_dict = {key: list(df_segment[df_segment.agent_type == key].track_id.unique()) \
                               for key in self.agent_type_names}
        agent_ids = [agent_id for agent_ids in agent_ids_dict.values() for agent_id in agent_ids]
        agent_types = [torch.LongTensor([i] * len(agent_ids_dict[agent_type])) \
                                 for i, agent_type in enumerate(self.agent_type_names)]
        agent_types = torch.cat(agent_types)
        agent_num = len(agent_ids)

        static_info = [df_segment[df_segment.track_id == agent].iloc[0] for agent in agent_ids]
        static_info = np.array([(v.length, v.width) for v in static_info], dtype=np.float32)

        df_segment['present'] = True
        frame_ids = sorted(df_segment.frame_id.unique())
        dense_index = pd.MultiIndex.from_product([agent_ids, frame_ids],
                                                 names=["track_id", "frame_id"])
        padding = pd.DataFrame(index=dense_index, data=dict(x=0.0, y=0.0, vx=0.0, \
                                                            vy=0.0, psi_rad=0.0, present=False))
        df_segment = df_segment.set_index(['track_id', 'frame_id']).reindex(dense_index)\
                               .combine_first(padding).reset_index()
        for track_id in df_segment['track_id'].unique():
            track = df_segment[(df_segment['track_id'] == track_id) & (df_segment['agent_type'])]
            df_segment.loc[(df_segment['track_id'] == track_id) & \
                           (df_segment['agent_type'].isna()), \
                           ['agent_type', 'agent_role', 'length', 'width']] = \
                track[['agent_type', 'agent_role', 'length', 'width']].values[0]
        track_id_dtype = df_segment['track_id'].dtype
        df_segment['track_id'] = pd.Categorical(df_segment['track_id'], agent_ids)
        df_segment = df_segment.sort_values(['track_id', 'frame_id'], ascending=[True, True], kind='mergesort')
        df_segment['track_id'] = df_segment['track_id'].astype(track_id_dtype)

        agents_state = df_segment[['x', 'y', 'psi_rad', 'vx', 'vy']].to_numpy(dtype=np.float32)
        agents_state = agents_state.reshape(agent_num, -1, 5)
        speed = np.linalg.norm(agents_state[..., 3:5], axis=-1, keepdims=True)
        agents_state = np.concatenate([agents_state[...,:3], speed], axis=-1)
        present_mask = df_segment['present'].to_numpy(dtype=np.bool_)
        present_mask = present_mask.reshape(agent_num, -1)

        item = {
            'agent_attributes': torch.tensor(static_info),
            'agent_states': torch.tensor(agents_state),
            'present_mask': torch.tensor(present_mask),
            'agent_types': agent_types,
            'road_mesh': self.road_meshes[location],
            'lane_mesh': self.lane_meshes[location],
        }
        return item

    @classmethod
    def collate(cls, items):
        batch = dict()
        n_agent_types = len(INTERACTIONDataset.agent_type_names)

        for k in items[0].keys():
            if k in ['agent_attributes', 'agent_states', 'present_mask']:
                batch[k] = torch.cat(
                    [torch.nn.utils.rnn.pad_sequence([x[items[j]['agent_types'] == i, ...] \
                    for (j, x) in enumerate([item[k] for item in items])], padding_value=0.0, batch_first=True)\
                     for i in range(n_agent_types)], dim=1
                )
            elif k == 'agent_types':
                batch[k] = torch.cat(
                    [torch.LongTensor([i] * max([torch.sum(item['agent_types'] == i) for item in items]))
                     for i in range(n_agent_types)]
                )
            elif k in ['road_mesh', 'lane_mesh']:
                batch[k] = BirdviewMesh.collate([item[k] for item in items])
            else:
                batch[k] = [item[k] for item in items]
        return batch


class BehaviorModel(nn.Module):
    def __init__(self, h_dim=64, action_dim=4, input_resolution=64, num_channels=3, fov=35):
        super(BehaviorModel, self).__init__()
        self.resolution = input_resolution
        self.fov = fov

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        intermediate_dim = int(np.prod(self.conv(torch.zeros(1, num_channels,
                                                             input_resolution, input_resolution)).shape))
        self.pred = nn.Sequential(
            nn.Linear(in_features=intermediate_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=h_dim),
            nn.ReLU(),
            nn.Linear(in_features=h_dim, out_features=action_dim),
            nn.Tanh(),
        )

        for weight in self.parameters():
            weight.data.normal_(0, 1e-1)

    def forward(self, birdviews):
        birdviews = ((birdviews / 255) - 0.5) / 0.5
        conv_fea = self.conv(birdviews)
        pred = self.pred(torch.flatten(conv_fea, start_dim=1))
        return pred


def ego_only_simulator(batch_data, simulator_cfg):
    agent_attributes = batch_data['agent_attributes']
    road_mesh = batch_data['road_mesh']
    lane_mesh = batch_data['lane_mesh']
    present_mask = batch_data['present_mask']
    replay_mask = torch.ones_like(batch_data['agent_types'], dtype=torch.bool)
    replay_mask[0] = False  # Ego agent is assumed to be first
    initial_state = batch_data['agent_states'][..., 0, :]

    agent_type_masks = {
        agent_type: batch_data['agent_types'] == i for (i, agent_type) \
                                    in enumerate(INTERACTIONDataset.agent_type_names)
    }

    kinematic_models = {}
    agent_sizes = {}
    agent_states = {}
    present_masks = {}
    initial_present_mask = {}
    replay_masks = {}
    for (agent_type, mask) in agent_type_masks.items():
        kinematic_models[agent_type] = SimpleKinematicModel()
        kinematic_models[agent_type].set_state(initial_state[:, mask])

        agent_sizes[agent_type] = agent_attributes[..., :2][:, mask]
        agent_states[agent_type] = batch_data['agent_states'][:, mask]
        present_masks[agent_type] = present_mask[:, mask]
        initial_present_mask[agent_type] = present_mask[:, mask][..., 0]
        replay_masks[agent_type] = replay_mask[mask]

    renderer = renderer_from_config(simulator_cfg.renderer, static_mesh=BirdviewMesh.concat([road_mesh, lane_mesh]))
    simulator = Simulator(cfg=simulator_cfg, road_mesh=road_mesh, kinematic_model=kinematic_models,
                          agent_size=agent_sizes, initial_present_mask=initial_present_mask, renderer=renderer)
    simulator = ReplayWrapper(simulator, npc_mask=replay_masks, agent_states=agent_states,
                              present_masks=present_masks)
    simulator = HomogeneousWrapper(simulator)
    return simulator


def predict_state(batch_data, model, simulator_cfg, teacher_forcing=False):
    agent_states = batch_data['agent_states']

    simulator = ego_only_simulator(batch_data, simulator_cfg)

    pred_states = [simulator.get_state()]
    for t in range(1, agent_states.shape[-2]):
        ego_bv = simulator.render_egocentric(res=Resolution(model.resolution, model.resolution), fov=model.fov)
        action = model(ego_bv.squeeze(1)).unsqueeze(1) # Temporarily remove the agent dim
        simulator.step(action)
        state = simulator.get_state()
        pred_states.append(state)
        if teacher_forcing:
            simulator.set_state(agent_states[..., :1, t, :])
    pred_states = torch.stack(pred_states, dim=-2)
    return pred_states


def validation_metrics(batch_data, model, simulator_cfg, teacher_forcing=False):
    with torch.no_grad():
        pred_state = predict_state(batch_data, model, simulator_cfg, teacher_forcing=teacher_forcing)
    distances = torch.linalg.norm(pred_state[..., 0, 1:, :2] -
                                  batch_data['agent_states'][..., 0, 1:, :2], ord=2, dim=-1)
    ade = torch.mean(distances, dim=-1)
    fde = distances[..., -1]
    return ade, fde


def train(args):
    device = args.device
    train_steps = args.train_steps

    simulator_cfg = TorchDriveConfig()

    train_dataset = INTERACTIONDataset(args.dataset_path, location_names=args.locations, split='train')
    train_dataloader = iter(cycle(torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                              batch_size=args.batch_size,
                                                              collate_fn=INTERACTIONDataset.collate)))
    val_dataset = INTERACTIONDataset(args.dataset_path, location_names=args.locations, split='val')
    val_dataset = val_dataset.subsample(num_segments=args.num_validation_samples)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False,
                                                 batch_size=10, collate_fn=INTERACTIONDataset.collate)
    print(f'Num. of training segments: {len(train_dataset)}')
    print(f'Num. of validation segments: {len(val_dataset)}')

    model = BehaviorModel(h_dim=args.h_dim, input_resolution=args.resolution, fov=args.fov).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    progress_bar = trange(train_steps, leave=True)
    for i in progress_bar:
        batch_data = next(train_dataloader)
        batch_data = to_device(batch_data, device)
        optimizer.zero_grad()
        pred_state = predict_state(batch_data, model, simulator_cfg, teacher_forcing=args.teacher_forcing)
        loss = F.mse_loss(pred_state[..., 0, 1:, :], batch_data['agent_states'][..., 0, 1:, :], reduction='none')
        loss = torch.mean(loss.sum(dim=-2)/40)
        loss.backward()
        optimizer.step()
        if i % args.validation_period == 0:
            val_ades = []
            val_fdes = []
            for val_batch_data in val_dataloader:
                val_batch_data = to_device(val_batch_data, device)
                val_ade, val_fde = validation_metrics(val_batch_data, model, simulator_cfg)
                val_ades.append(val_ade)
                val_fdes.append(val_fde)
            val_ades = torch.cat(val_ades, dim=0).mean().item()
            val_fdes = torch.cat(val_fdes, dim=0).mean().item()
        progress_bar.set_description(f"Train: Loss={loss.item():.2f} - Val: ADE={val_ades:.2f}, FDE={val_fdes:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the INTERACTION dataset folder.')
    parser.add_argument('--locations', nargs='+', type=str, default=None, help='Locations to train with.')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps.')
    parser.add_argument('--num_validation_samples', type=int, default=10, help='Number of validation samples.')
    parser.add_argument('--validation_period', type=int, default=10, help='How often to run validation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run training (cpu/cuda).')
    parser.add_argument('--resolution', type=int, default=64, help='Birdview resolution.')
    parser.add_argument('--fov', type=int, default=35, help='Birdview field-of-view.')
    parser.add_argument('--h_dim', type=int, default=64, help='Model hidden dimensionality.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--teacher_forcing', type=bool, default=False, help='Use teacher forcing during training.')

    args = parser.parse_args()

    train(args)
