import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import numpy as np
import time
import datetime
from common.utils import centering_image, denorm_image
from common.dataset import TrainDataProvider
from common.function import init_embedding
from common.models import Encoder, Decoder, Discriminator, Generator
from matplotlib import pyplot as plt
from tqdm import tqdm
from font_maker import bmp_to_svg, make_font, run_make_font


batch_size = 16
IMG_SIZE = 128
EMBEDDING_DIM = 128
FONTS_NUM = 25
EMBEDDING_NUM = 100

data_dir = './static/handwritings/realimg_with_srcfont'
embedding_dir = data_dir
save_path = './static/fixed_real_fake/'
os.makedirs(save_path, exist_ok=True)
to_model_path = './model_checkpoint2/'
os.makedirs(to_model_path, exist_ok=True)
from_model_path = './model_checkpoint/'
restore = ['200-0228-19_52-Encoder.pkl', '200-0228-19_52-Decoder.pkl', '200-0228-19_52-Discriminator.pkl']
max_epoch = 10000
schedule = 500
lr = 0.001
log_step = 10
sample_step = 10
resize_fix = 90
BATCH_SIZE = 16

save_dir = data_dir

embeddings = init_embedding(EMBEDDING_NUM, EMBEDDING_DIM)
torch.save(embeddings, os.path.join(save_dir, 'EMBEDDINGS.pkl'))



# training
def train(max_epoch, schedule, save_path, to_model_path, lr=0.001, \
          log_step=100, sample_step=350, fine_tune=False, flip_labels=False, \
          restore=None, from_model_path='', GPU=True, freeze_encoder=False,
          with_charid=False, resize_fix=90, gan_loss=True):
    # Fine Tuning coefficient
    if not fine_tune:
        L1_penalty, Lconst_penalty = 100, 15
    else:
        L1_penalty, Lconst_penalty = 500, 1000

    fixed_source = torch.load(os.path.join(save_dir, 'fixed_source.pkl'))
    fixed_target = torch.load(os.path.join(save_dir, 'fixed_target.pkl'))
    fixed_label = torch.load(os.path.join(save_dir, 'fixed_label.pkl'))

    data_provider = TrainDataProvider(data_dir)
    total_batches = data_provider.compute_total_batch_num(BATCH_SIZE)
    print("total batches:", total_batches)

    # Get Models
    En = Encoder()
    De = Decoder()
    D = Discriminator(category_num=FONTS_NUM)
    if GPU:
        En.cuda()
        De.cuda()
        D.cuda()
        embeddings.cuda()

    # Use pre-trained Model
    # restore에 [encoder_path, decoder_path, discriminator_path] 형태로 인자 넣기
    if restore:
        encoder_path, decoder_path, discriminator_path = restore
        prev_epoch = int(encoder_path.split('-')[0])
        En.load_state_dict(torch.load(os.path.join(from_model_path, encoder_path)))
        De.load_state_dict(torch.load(os.path.join(from_model_path, decoder_path)))
        D.load_state_dict(torch.load(os.path.join(from_model_path, discriminator_path)))
        print("%d epoch trained model has restored" % prev_epoch)
    else:
        prev_epoch = 0
        print("New model training start")

    # L1 loss, binary real/fake loss, category loss, constant loss
    if GPU:
        l1_criterion = nn.L1Loss(size_average=True).cuda()
        bce_criterion = nn.BCEWithLogitsLoss(size_average=True).cuda()
        mse_criterion = nn.MSELoss(size_average=True).cuda()
    else:
        l1_criterion = nn.L1Loss(size_average=True)
        bce_criterion = nn.BCEWithLogitsLoss(size_average=True)
        mse_criterion = nn.MSELoss(size_average=True)

    # optimizer
    if freeze_encoder:
        G_parameters = list(De.parameters())
    else:
        G_parameters = list(En.parameters()) + list(De.parameters())
    g_optimizer = torch.optim.Adam(G_parameters, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), betas=(0.5, 0.999))

    # losses lists
    l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()

    count = 0
    for epoch in range(max_epoch):
        if (epoch + 1) % schedule == 0:
            updated_lr = max(lr / 2, 0.0002)
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = updated_lr
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = updated_lr
            if lr != updated_lr:
                print("decay learning rate from %.5f to %.5f" % (lr, updated_lr))
            lr = updated_lr

        train_batch_iter = data_provider.get_train_iter(BATCH_SIZE, with_charid=with_charid)
        for i, batch in enumerate(train_batch_iter):
            if with_charid:
                font_ids, char_ids, batch_images = batch
            else:
                font_ids, batch_images = batch
            embedding_ids = font_ids

            if GPU:
                batch_images = batch_images.cuda()
            if flip_labels:
                np.random.shuffle(embedding_ids)

            # target / source images
            real_target = batch_images[:, 0, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])
            real_source = batch_images[:, 1, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])

            # centering
            for idx, (image_S, image_T) in enumerate(zip(real_source, real_target)):
                image_S = image_S.cpu().detach().numpy().reshape(IMG_SIZE, IMG_SIZE)
                image_S = centering_image(image_S, resize_fix=90)
                real_source[idx] = torch.tensor(image_S).view([1, IMG_SIZE, IMG_SIZE])
                image_T = image_T.cpu().detach().numpy().reshape(IMG_SIZE, IMG_SIZE)
                image_T = centering_image(image_T, resize_fix=resize_fix)
                real_target[idx] = torch.tensor(image_T).view([1, IMG_SIZE, IMG_SIZE])

            # generate fake image form source image
            fake_target, encoded_source, encoder_layers = Generator(real_source, En, De, embeddings, embedding_ids,
                                                                    GPU=GPU)

            real_TS = torch.cat([real_source, real_target], dim=1)
            fake_TS = torch.cat([real_source, fake_target], dim=1)

            # Scoring with Discriminator
            fake_score, fake_score_logit, fake_cat_logit = D(fake_TS)

            # Get encoded fake image to calculate constant loss
            encoded_fake = En(fake_target)[0]
            const_loss = Lconst_penalty * mse_criterion(encoded_source, encoded_fake)

            # category loss
            real_category = torch.from_numpy(np.eye(FONTS_NUM)[embedding_ids]).float()
            if GPU:
                real_category = real_category.cuda()
            fake_category_loss = bce_criterion(fake_cat_logit, real_category)

            # labels
            if GPU:
                one_labels = torch.ones([BATCH_SIZE, 1]).cuda()
                zero_labels = torch.zeros([BATCH_SIZE, 1]).cuda()
            else:
                one_labels = torch.ones([BATCH_SIZE, 1])
                zero_labels = torch.zeros([BATCH_SIZE, 1])

            # L1 loss between real and fake images
            l1_loss = L1_penalty * l1_criterion(real_target, fake_target)

            # cheat loss for generator to fool discriminator
            cheat_loss = bce_criterion(fake_score_logit, one_labels)

            # g_loss, d_loss
            g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss

            # train Generator
            En.zero_grad()
            De.zero_grad()

            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            fake_TS = torch.cat([real_source, fake_target.detach().clone()], dim=1)
            real_score, real_score_logit, real_cat_logit = D(real_TS)
            fake_score, fake_score_logit, fake_cat_logit = D(fake_TS.detach().clone())
            # binary loss for discriminator
            if gan_loss:
                real_binary_loss = bce_criterion(real_score_logit, one_labels)
                fake_binary_loss = bce_criterion(fake_score_logit, zero_labels)
                binary_loss = real_binary_loss + fake_binary_loss
            # category loss for discriminator
            fake_category_loss = bce_criterion(fake_cat_logit, real_category)
            real_category_loss = bce_criterion(real_cat_logit, real_category)
            category_loss = 0.5 * (real_category_loss + fake_category_loss)
            if gan_loss:
                d_loss = binary_loss + category_loss
            else:
                d_loss = category_loss
            # train Discriminator
            D.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # loss data
            l1_losses.append(l1_loss.data)
            const_losses.append(const_loss.data)
            category_losses.append(category_loss.data)
            d_losses.append(d_loss.data)
            g_losses.append(g_loss.data)

            # logging
            if (i + 1) % log_step == 0:
                time_ = time.time()
                time_stamp = datetime.datetime.fromtimestamp(time_).strftime('%H:%M:%S')
                log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, d_loss: %.4f, g_loss: %.4f' % \
                             (int(prev_epoch) + epoch + 1, int(prev_epoch) + max_epoch, i + 1, total_batches, \
                              l1_loss.item(), d_loss.item(), g_loss.item())
                print(time_stamp, log_format)

            # save image
            if (i + 1) % sample_step == 0:
                fixed_fake_images = Generator(fixed_source, En, De, embeddings, fixed_label, GPU=GPU)[0]
                save_image(denorm_image(fixed_fake_images.data), \
                           os.path.join(save_path, 'fake_samples-%d-%d.png' % (int(prev_epoch) + epoch + 1, i + 1)), \
                           nrow=8)

        if (epoch + 1) % 1000 == 0:
            now = datetime.datetime.now()
            now_date = now.strftime("%m%d")
            now_time = now.strftime('%H:%M')
            torch.save(En.state_dict(), os.path.join(to_model_path, '%d-%s-%s-Encoder.pkl' \
                                                     % (int(prev_epoch) + epoch + 1, now_date, now_time)))
            torch.save(De.state_dict(), os.path.join(to_model_path, '%d-%s-%s-Decoder.pkl' % \
                                                     (int(prev_epoch) + epoch + 1, now_date, now_time)))
            torch.save(D.state_dict(), os.path.join(to_model_path, '%d-%s-%s-Discriminator.pkl' % \
                                                    (int(prev_epoch) + epoch + 1, now_date, now_time)))

    # save model
    total_epoch = int(prev_epoch) + int(max_epoch)
    end = datetime.datetime.now()
    end_date = end.strftime("%m%d")
    end_time = end.strftime('%H:%M')
    torch.save(En.state_dict(), os.path.join(to_model_path, \
                                             '%d-%s-%s-Encoder.pkl' % (total_epoch, end_date, end_time)))
    torch.save(De.state_dict(), os.path.join(to_model_path, \
                                             '%d-%s-%s-Decoder.pkl' % (total_epoch, end_date, end_time)))
    torch.save(D.state_dict(), os.path.join(to_model_path, \
                                            '%d-%s-%s-Discriminator.pkl' % (total_epoch, end_date, end_time)))
    losses = [l1_losses, const_losses, category_losses, d_losses, g_losses]
    torch.save(losses, os.path.join(to_model_path, '%d-losses.pkl' % max_epoch))

    return l1_losses, const_losses, category_losses, d_losses, g_losses

def char_list():
    with open('hangul-11172.txt', 'r') as f:
        chars = f.readlines()
    return chars

def make_bmp(font_name="아무노래"):
    chars = char_list()[0][:-1]
    print(len(chars))
    embeddings = torch.load(os.path.join(embedding_dir, 'EMBEDDINGS.pkl'))
    restores = os.listdir(to_model_path)
    restores.sort()
    restore_En, restore_De, restore_Di = list(filter(lambda x:'Encoder' in x, restores))[-1], list(filter(lambda x:'Decoder' in x, restores))[-1], list(filter(lambda x:'Discri' in x, restores))[-1]
    restore = [restore_En, restore_De, restore_Di]
    print(restore)
    with torch.no_grad():
        En = Encoder().cuda()
        De = Decoder().cuda()
        D = Discriminator(category_num=25).cuda()
        encoder_path, decoder_path, discriminator_path = restore
        prev_epoch = int(encoder_path.split('-')[0])
        En.load_state_dict(torch.load(os.path.join(to_model_path, encoder_path)))
        De.load_state_dict(torch.load(os.path.join(to_model_path, decoder_path)))
        D.load_state_dict(torch.load(os.path.join(to_model_path, discriminator_path)))
        print("%d epoch trained model has restored" % prev_epoch)
    from common.utils import chars_to_ids
    selected_chars = chars_to_ids(chars)
    fixed_char_ids = np.array(selected_chars)
    print(f"len(os.listdir('hangul-dataset-11172')) : {len(os.listdir('hangul-dataset-11172'))}")
    
    font_filter = [0]
    from_dir = 'hangul-dataset-11172/' # hangul-dataset-11172 must need 11172 data!
    save_path = 'static/handwritings/all_latters.obj'
    from get_data.package import pickle_examples, pickle_interpolation_data
    pickle_interpolation_data(from_dir, save_path, fixed_char_ids, font_filter)

    BATCH_SIZE = 10
    IMG_SIZE = 128

    data_provider = TrainDataProvider('static/handwritings/', val=False, train_name='all_latters.obj', \
                                              filter_by_font=font_filter)
    total_batches = data_provider.compute_total_batch_num(BATCH_SIZE)
    print("total batches:", total_batches)

    train_batch_iter = data_provider.get_train_iter(BATCH_SIZE, shuffle=False,with_charid=True)

    results = {i: {j: None for j in fixed_char_ids} for i in font_filter}
    GPU=True

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    En.eval()
    De.eval()
    idx = 0
    save_result = f'static/results/handwiting_fonts_{font_name}'
    os.makedirs(save_result, exist_ok=True)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_batch_iter)):
            font_ids, char_ids, batch_images = batch
            batch_images = batch_images.cuda()
            embedding_ids = font_ids
            # target / source images
            real_targets = batch_images[:, 0, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])
            real_sources = batch_images[:, 1, :, :].view([BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE])
                                                                                
            # generate fake image
            fake_targets, encoded_sources, _ = Generator(real_sources, En, De, embeddings, embedding_ids, GPU=GPU, encode_layers=False)

            # save fake images by filtering with fonts
            for data in zip(font_ids, char_ids, encoded_sources, real_sources, real_targets, fake_targets):
                fontid, charid, encoded, real_S, real_T, fake_T = data
                real_S = real_S.cpu().detach().numpy()
                real_S = centering_image(real_S.reshape(128, 128), resize_fix=90)
                real_T = real_T.cpu().detach().numpy()
                real_T = centering_image(real_T.reshape(128, 128), resize_fix=90)
                fake_T = fake_T.cpu().detach().numpy()
                fake_T = centering_image(fake_T.reshape(128, 128), resize_fix=90)
                # (0)encoded, (1)layers, (2)real_source, (3)real_target, (4)fake_target 순서대로 저장
                results[fontid][charid] = [real_S, real_T, fake_T]
                plt.imsave(f'{save_result}/{charid}.bmp', fake_T, format='bmp', cmap='gray')
                idx += 1
            fake_targets = None
            encoded_sources = None
            gc.collect()
            torch.cuda.empty_cache()
    print(len(os.listdir(save_result)))
    return save_result

def bmp_convert_and_makefont(save_bmp_path):
    print('start saving svgs!')
    save_svgs = bmp_to_svg(save_bmp_path)
    print('saving svg complete!')
    fontname = make_font(save_svgs)
    print('saving json for makeing font complete!') 
    run_make_font(fontname)
    print('total complete')

def run_train(epoch):
    losses = train(max_epoch=epoch, schedule=schedule, save_path=save_path,
                   from_model_path=from_model_path, to_model_path=to_model_path, restore=restore,
                   log_step=log_step, sample_step=sample_step, lr=lr, freeze_encoder=True,
                   with_charid=True, resize_fix=resize_fix, gan_loss=False)

if __name__ == '__main__':
    run_train()
