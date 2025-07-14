import time
import logging

import torch

from utils import AverageMeter

logger = logging.getLogger(__name__)

def train(config, train_loader, model, vgg, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    R_loss_average = AverageMeter()
    V_loss_average = AverageMeter()
    P_loss_average = AverageMeter()
    E_loss_average = AverageMeter()
    S_loss_average = AverageMeter()
    loss_average = AverageMeter()

    model.train()
    model.module.backbone1.eval()
    vgg.eval()

    end = time.time()

    for i, meta in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = meta['Img'].cuda()
        input_small = meta['Small_input'].cuda()

        Presence_loss, Reconstruction_loss, Variation_loss, Entropy_loss, Semantic_loss = model(input, input_small, vgg, True)

        loss = Reconstruction_loss + 0.5 * (Variation_loss + Entropy_loss) + 1.0 * Presence_loss + 0.25 * Semantic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        loss_average.update(loss.item(), input.size(0))
        R_loss_average.update(Reconstruction_loss.item(), input.size(0))
        V_loss_average.update(Variation_loss.item(), input.size(0))
        P_loss_average.update(Presence_loss.item(), input.size(0))
        E_loss_average.update(Entropy_loss.item(), input.size(0))
        S_loss_average.update(Semantic_loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'R_loss: {R_loss.val:.5f} ({R_loss.avg:.5f})\t' \
                  'V_loss: {V_loss.val:.5f} ({V_loss.avg:.5f})\t' \
                  'P_loss: {P_loss.val:.5f} ({P_loss.avg:.5f})\t' \
                  'E_loss: {E_loss.val:.5f} ({E_loss.avg:.5f})\t' \
                  'S_loss: {S_loss.val:.5f} ({S_loss.avg:.5f})\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, V_loss=V_loss_average,
                P_loss=P_loss_average, S_loss=S_loss_average,
                R_loss=R_loss_average, E_loss=E_loss_average,
                loss=loss_average)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', loss_average.val, global_steps)
            writer.add_scalar('V_loss', V_loss_average.val, global_steps)
            writer.add_scalar('R_loss', R_loss_average.val, global_steps)
            writer.add_scalar('P_loss', P_loss_average.val, global_steps)
            writer.add_scalar('E_loss', E_loss_average.val, global_steps)
            writer.add_scalar('S_loss', S_loss_average.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

