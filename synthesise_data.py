import torch
import torch.nn as nn
from latent_code_generation import get_w_labelled, get_noise, get_w_unlabelled, get_noise_custom
import numpy as np
from PIL import Image
import os


def generate_examples(gen, epoch, cross_level_test, log_resolution, num_layers, mapping_network_labelled,
                      mapping_network_unlabelled, w_dim, n=100, num_classes=10, device='cpu',
                      save_path="./crossloc_examples", training_mode="cross", use_custom_nets=False):
    gen.eval()

    with torch.no_grad():
        if training_mode in ["lo", "mixed", "cross"]:
            # generate labelled images
            labels = torch.tensor([n for n in range(num_classes)] * int(n / num_classes))
            labels = labels.to(torch.int64)
            labels = nn.functional.one_hot(labels, num_classes)

            labels = labels.to(device)

            w = get_w_labelled(n, labels, mapping_network_labelled, w_dim, device, num_layers)
            if use_custom_nets:
                noise = get_noise_custom(n, device, log_resolution, num_layers)
            else:
                noise = get_noise(n, device, log_resolution, num_layers)
            img = gen(w, noise)

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (0, 2, 3, 1))

            r = []

            for i in range(0, n, int(n / num_classes)):
                r.append(np.concatenate(img[i:i + int(n / num_classes)], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.squeeze(np.uint8(c1 * 255)))

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            x.save(save_path + f'/epoch{epoch}.png')

        if training_mode in ["mixed", "cross"]:
            # generate cross/mixed images
            w_c_labelled = get_w_labelled(n, labels, mapping_network_labelled, w_dim, device, num_layers)
            w_c_unlabelled = get_w_unlabelled(n, mapping_network_unlabelled, w_dim, device, num_layers)

            tt = cross_level_test

            w_cross = torch.cat((w_c_labelled[:tt, :, :], w_c_unlabelled[:(num_layers - tt + 1), :, :]), axis=0)

            if use_custom_nets:
                noise = get_noise_custom(n, device, log_resolution, num_layers)
            else:
                noise = get_noise(n, device, log_resolution, num_layers)

            img = gen(w_cross, noise)

            img = img.detach().cpu().numpy()

            img = np.transpose(img, (0, 2, 3, 1))

            r = []

            for i in range(0, n, int(n / num_classes)):
                r.append(np.concatenate(img[i:i + int(n / num_classes)], axis=1))

            c1 = np.concatenate(r, axis=0)
            c1 = np.clip(c1, 0.0, 1.0)
            x = Image.fromarray(np.squeeze(np.uint8(c1 * 255)))

            x.save(save_path + f'/epoch{epoch}_c.png')
