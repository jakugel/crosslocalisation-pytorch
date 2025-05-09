from random import random
from synthesise_data import generate_examples
import torch
import numpy as np
import torch.nn as nn
from latent_code_generation import get_w_unlabelled, get_w_labelled, get_noise, get_noise_custom
from penalties import gradient_penalty
import os


def train_fn(
        critic_labelled, critic_unlabelled,
        gen, mapnetl, mapnetu,
        path_length_penalty,
        images_labelled, labels, images_unlabelled,
        opt_critic_labelled, opt_critic_unlabelled,
        opt_gen,
        opt_mapping_network_labelled, opt_mapping_network_unlabelled, num_epochs, batch_size, device, num_classes,
        img_size, mixing_prob, logres, w_dim, lambda_gp, num_gen_layers, training_mode, loss, cross_level_train,
        cross_level_test, use_custom_nets, save_models, save_models_freq, save_models_dir, save_images,
        save_images_freq, save_images_dir
):

    if save_models:
        if not os.path.exists(save_models_dir):
            os.makedirs(save_models_dir)

    for epoch in range(num_epochs):
        if training_mode in ["lo", "mixed", "cross"]:
            ########## LABELLED DATA ITERATION ############
            loss_critic_labelled, loss_gen_labelled = labelled_training_step(images_labelled, batch_size, device, labels, num_classes, img_size, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen, critic_labelled, loss, lambda_gp, epoch, opt_critic_labelled,
                           path_length_penalty, opt_gen, opt_mapping_network_labelled, use_custom_nets)

        if training_mode in ["uo", "mixed", "cross"]:
            ########## UNLABELLED DATA ITERATION ############
            loss_critic_unlabelled, loss_gen_unlabelled = unlabelled_training_step(images_unlabelled, batch_size, device, mixing_prob, num_gen_layers, mapnetu, w_dim, logres,
                             gen, critic_unlabelled, loss, lambda_gp, opt_critic_unlabelled, epoch, path_length_penalty,
                             opt_gen, opt_mapping_network_unlabelled, use_custom_nets)

        if training_mode == 'cross':
            ######## CROSS ITERATION ###########
            loss_critic_cross, loss_gen_cross = cross_training_step(images_unlabelled, images_labelled, batch_size, device, labels, num_classes, num_gen_layers,
                        mapnetl, w_dim, mapnetu, logres, gen, critic_unlabelled, loss, lambda_gp, opt_critic_unlabelled,
                        epoch, path_length_penalty, opt_gen, opt_mapping_network_unlabelled, opt_mapping_network_labelled,
                                                                    cross_level_train, use_custom_nets)

        ######### Print losses every 10 epochs ##########
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
            if training_mode in ["lo", "mixed", "cross"]:
                print(f"Loss Critic Labelled: {loss_critic_labelled.item():.4f}")
                print(f"Loss Gen Labelled: {loss_gen_labelled.item():.4f}")

            if training_mode in ["uo", "mixed", "cross"]:
                print(f"Loss Critic Unlabelled: {loss_critic_unlabelled.item():.4f}")
                print(f"Loss Gen Unlabelled: {loss_gen_unlabelled.item():.4f}")

            if training_mode == "cross":
                print(f"Loss Critic Cross: {loss_critic_cross.item():.4f}")
                print(f"Loss Gen Cross: {loss_gen_cross.item():.4f}")

            print("-" * 50)

        if epoch % save_images_freq == 0 and save_images:
            generate_examples(gen, epoch, cross_level_test, log_resolution=logres, num_layers=num_gen_layers, mapping_network_labelled=mapnetl,
                              mapping_network_unlabelled=mapnetu, w_dim=w_dim, device=device, training_mode=training_mode,
                              use_custom_nets=use_custom_nets, save_path=save_images_dir)
            gen.train()

        if epoch % save_models_freq == 0 and save_models:
            torch.save(critic_labelled.state_dict(), save_models_dir + f"/criticl_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/criticl_{epoch}.pth")
            torch.save(critic_unlabelled.state_dict(), save_models_dir + f"/criticu_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/criticu_{epoch}.pth")
            torch.save(gen.state_dict(), save_models_dir + f"/gen_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/gen_{epoch}.pth")
            torch.save(mapnetl.state_dict(), save_models_dir + f"/mapnetl_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/mapnetl_{epoch}.pth")
            torch.save(mapnetu.state_dict(), save_models_dir + f"/mapnetu_{epoch}.pth")
            print('saved model at ' + save_models_dir + f"/mapnetu_{epoch}.pth")


def labelled_training_step(images_labelled, batch_size, device, labels, num_classes, img_size, mixing_prob, num_gen_layers,
                           mapnetl, w_dim, logres, gen, critic_labelled, loss, lambda_gp, epoch, opt_critic_labelled,
                           path_length_penalty, opt_gen, opt_mapping_network_labelled, use_custom_nets):
    # get real image batch
    idx1 = np.random.randint(0, images_labelled.shape[0], batch_size)

    # images
    batch_images_l = images_labelled[idx1].astype('float32') / 255.0

    real_labelled = torch.Tensor(batch_images_l)
    real_labelled = real_labelled.to(device)

    # labels
    batch_labels = labels[idx1]

    label = torch.Tensor(batch_labels)
    label = label.to(torch.int64)
    label = nn.functional.one_hot(label, num_classes)

    # labels (image size)
    labelimg = label.view(-1, num_classes, 1, 1)
    labelimg = labelimg.repeat(1, 1, img_size, img_size)

    labelimg = labelimg.to(device)
    label = label.to(device)
    cur_batch_size = real_labelled.shape[0]

    # style mixing
    if random() < mixing_prob:
        # mixing
        w1_labelled = get_w_labelled(cur_batch_size, label, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)
        w2_labelled = get_w_labelled(cur_batch_size, label, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                     w_dim=w_dim)

        tt = int(random() * num_gen_layers)  # style level to crossover

        w_labelled = torch.cat((w1_labelled[:tt, :, :], w2_labelled[:(num_gen_layers - tt + 1), :, :]), axis=0)
    else:
        # no mixing
        w_labelled = get_w_labelled(cur_batch_size, label, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                    w_dim=w_dim)
    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_labelled = gen(w_labelled, noise)  # get fake labelled images
    critic_fake_labelled = critic_labelled(fake_labelled.detach(), labelimg)  # compute critic scores for fake images

    critic_real_labelled = critic_labelled(real_labelled, labelimg)  # compute critic scores for real images

    # compute gradient penalty
    gp_labelled = gradient_penalty(critic_labelled, real_labelled, fake_labelled, labelimg, device=device)

    # compute full critic loss using labelled image training
    if loss == "wgan":
        loss_critic_labelled = (
                -(torch.mean(critic_real_labelled) - torch.mean(critic_fake_labelled))
                + lambda_gp * gp_labelled
                + (0.001 * torch.mean(critic_real_labelled ** 2))
        )
    elif loss == "hinge":
        loss_critic_labelled = (
                torch.mean(torch.relu(1 + critic_real_labelled) + torch.relu(1 - critic_fake_labelled))
                + lambda_gp * gp_labelled
        )

    critic_labelled.zero_grad()
    loss_critic_labelled.backward()
    opt_critic_labelled.step()

    gen_fake_labelled = critic_labelled(fake_labelled, labelimg)

    if loss == "wgan":
        loss_gen_labelled = -torch.mean(gen_fake_labelled)
    elif loss == "hinge":
        loss_gen_labelled = torch.mean(gen_fake_labelled)

    if epoch % 16 == 0:
        plp_labelled = path_length_penalty(w_labelled, fake_labelled)
        if not torch.isnan(plp_labelled):
            loss_gen_labelled = loss_gen_labelled + plp_labelled

    mapnetl.zero_grad()
    gen.zero_grad()
    loss_gen_labelled.backward()
    opt_gen.step()
    opt_mapping_network_labelled.step()

    return loss_critic_labelled, loss_gen_labelled


def unlabelled_training_step(images_unlabelled, batch_size, device, mixing_prob, num_gen_layers, mapnetu, w_dim, logres,
                             gen, critic_unlabelled, loss, lambda_gp, opt_critic_unlabelled, epoch, path_length_penalty,
                             opt_gen, opt_mapping_network_unlabelled, use_custom_nets):
    idx2 = np.random.randint(0, images_unlabelled.shape[0], batch_size)

    batch_images_u = images_unlabelled[idx2].astype('float32') / 255.0

    real_unlabelled = torch.Tensor(batch_images_u)
    real_unlabelled = real_unlabelled.to(device)
    cur_batch_size = real_unlabelled.shape[0]

    if random() < mixing_prob:
        w1_unlabelled = get_w_unlabelled(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                         w_dim=w_dim)
        w2_unlabelled = get_w_unlabelled(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                         w_dim=w_dim)

        tt = int(random() * num_gen_layers)

        w_unlabelled = torch.cat((w1_unlabelled[:tt, :, :], w2_unlabelled[:(num_gen_layers - tt + 1), :, :]), axis=0)

    else:
        w_unlabelled = get_w_unlabelled(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                        w_dim=w_dim)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_unlabelled = gen(w_unlabelled, noise)
    critic_fake_unlabelled, _ = critic_unlabelled(fake_unlabelled.detach())
    critic_real_unlabelled, _ = critic_unlabelled(real_unlabelled)
    gp_unlabelled = gradient_penalty(critic_unlabelled, real_unlabelled, fake_unlabelled,
                                     device=device)

    if loss == "wgan":
        loss_critic_unlabelled = (
                -(torch.mean(critic_real_unlabelled) - torch.mean(critic_fake_unlabelled))
                + lambda_gp * gp_unlabelled
                + (0.001 * torch.mean(critic_real_unlabelled ** 2))
        )
    elif loss == "hinge":
        loss_critic_unlabelled = (
                torch.mean(torch.relu(1 + critic_real_unlabelled) + torch.relu(1 - critic_fake_unlabelled))
                + lambda_gp * gp_unlabelled
        )

    critic_unlabelled.zero_grad()
    loss_critic_unlabelled.backward()
    opt_critic_unlabelled.step()

    gen_fake_unlabelled, _ = critic_unlabelled(fake_unlabelled)

    if loss == "wgan":
        loss_gen_unlabelled = -torch.mean(gen_fake_unlabelled)
    elif loss == "hinge":
        loss_gen_unlabelled = torch.mean(gen_fake_unlabelled)

    if epoch % 16 == 0:
        plp_unlabelled = path_length_penalty(w_unlabelled, fake_unlabelled)
        if not torch.isnan(plp_unlabelled):
            loss_gen_unlabelled = loss_gen_unlabelled + plp_unlabelled

    mapnetu.zero_grad()
    gen.zero_grad()
    loss_gen_unlabelled.backward()
    opt_gen.step()
    opt_mapping_network_unlabelled.step()

    return loss_critic_unlabelled, loss_gen_unlabelled


def cross_training_step(images_unlabelled, images_labelled, batch_size, device, labels, num_classes, num_gen_layers,
                        mapnetl, w_dim, mapnetu, logres, gen, critic_unlabelled, loss, lambda_gp, opt_critic_unlabelled,
                        epoch, path_length_penalty, opt_gen, opt_mapping_network_unlabelled, opt_mapping_network_labelled,
                        cross_level_train, use_custom_nets):
    idx3 = np.random.randint(0, images_unlabelled.shape[0], batch_size)
    idx4 = np.random.randint(0, images_labelled.shape[0], batch_size)

    batch_images_c = images_unlabelled[idx3].astype('float32') / 255.0

    real_cross = torch.Tensor(batch_images_c)
    real_cross = real_cross.to(device)

    batch_labels_cross = labels[idx4]

    labelcross = torch.Tensor(batch_labels_cross)
    labelcross = labelcross.to(torch.int64)
    labelcross = nn.functional.one_hot(labelcross, num_classes)

    labelcross = labelcross.to(device)
    cur_batch_size = real_cross.shape[0]

    w_c_labelled = get_w_labelled(cur_batch_size, labelcross, device=device, num_layers=num_gen_layers, mapnet=mapnetl,
                                  w_dim=w_dim)
    w_c_unlabelled = get_w_unlabelled(cur_batch_size, device=device, num_layers=num_gen_layers, mapnet=mapnetu,
                                      w_dim=w_dim)

    tt = cross_level_train

    w_cross = torch.cat((w_c_labelled[:tt, :, :], w_c_unlabelled[:(num_gen_layers - tt + 1), :, :]), axis=0)

    if use_custom_nets:
        noise = get_noise_custom(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)
    else:
        noise = get_noise(cur_batch_size, device=device, logres=logres, num_layers=num_gen_layers)

    fake_cross = gen(w_cross, noise)
    _, critic_fake_cross = critic_unlabelled(fake_cross.detach())

    _, critic_real_cross = critic_unlabelled(real_cross)
    gp_cross = gradient_penalty(critic_unlabelled, real_cross, fake_cross, cross=True, device=device)

    if loss == 'wgan':
        loss_critic_cross = (
                -(torch.mean(critic_real_cross) - torch.mean(critic_fake_cross))
                + lambda_gp * gp_cross
                + (0.001 * torch.mean(critic_real_cross ** 2))
        )
    elif loss == 'hinge':
        loss_critic_cross = (
                torch.mean(torch.relu(1 + critic_real_cross) + torch.relu(1 - critic_fake_cross))
                + lambda_gp * gp_cross
        )

    critic_unlabelled.zero_grad()
    loss_critic_cross.backward()
    opt_critic_unlabelled.step()

    _, gen_fake_cross = critic_unlabelled(fake_cross)

    if loss == 'wgan':
        loss_gen_cross = -torch.mean(gen_fake_cross)
    elif loss == 'hinge':
        loss_gen_cross = torch.mean(gen_fake_cross)

    if epoch % 16 == 0:
        plp_cross = path_length_penalty(w_cross, fake_cross)
        if not torch.isnan(plp_cross):
            loss_gen_cross = loss_gen_cross + plp_cross

    mapnetl.zero_grad()
    mapnetu.zero_grad()
    gen.zero_grad()
    loss_gen_cross.backward()
    opt_gen.step()
    opt_mapping_network_labelled.step()
    opt_mapping_network_unlabelled.step()

    return loss_critic_cross, loss_gen_cross