import sys
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def actions_to_arr(actions):
    clip = lambda x: min(max(0, x), 1)
    result = [[clip(x["throttle"]) - clip(x["brake"]), x["steer"]] for x in actions]
    return torch.Tensor(result)


if __name__ == "__main__":
    data_dir = Path('/home/casa/projects/bruno/carla-diffuser-bc/bet_data_release/carla')

    vids = []
    acts = []
    for d in tqdm(list(data_dir.glob("*"))):
        vid_path = str(d / "0.mp4")
        act_path = str(d / "actions.json")
        vid, _, metadata = torchvision.io.read_video(str(vid_path))
        with open(act_path) as f:
            act = json.load(f)
            act = actions_to_arr(act)
        vid = rearrange(vid, "t h w c -> t c h w")
        act = act[(len(act) - len(vid)) :]
        assert vid.shape[0] == act.shape[0]

        vids.append(vid)
        acts.append(act)

    lengths = [x.shape[0] for x in vids]
    vids = pad_sequence(vids, batch_first=True, padding_value=0)
    acts = pad_sequence(acts, batch_first=True, padding_value=0)

    print("Saving to %s. This might take a while..." % data_dir)
    torch.save(lengths, data_dir / "seq_lengths.pth")
    torch.save(vids, data_dir / "all_observations.pth")
    torch.save(acts, data_dir / "all_actions_pm1.pth")
    print("Done.")

