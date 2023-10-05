import logging
import os
import time
import torch
import torch.nn as nn
from torch.cuda import amp

def do_train(args,
             model,
             criterion,
             train_dataloader,
             eval_dataloader,
             optimizer,
             scheduler,
             local_rank):
    log_period = args.log_period
    checkpoint_period = args.checkpoint_period
    eval_period = args.eval_period

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = args.max_epochs

    logger = logging.getLogger("simclr")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device == "cuda":
        model.to(local_rank)

    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        for n_iter, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            avg_loss = total_loss / len(train_dataloader)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                base_lr = scheduler._get_lr(epoch)[0]
                logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_dataloader), avg_loss, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        scheduler.step(epoch)
        logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch * (n_iter + 1), train_dataloader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, args.model_name + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            total_loss = 0
            for n_iter, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    x0, x1 = batch[0]
                    x0 = x0.to(device)
                    x1 = x1.to(device)
                    z0 = model(x0)
                    z1 = model(x1)
                    loss = criterion(z0, z1)
                    total_loss += loss.detach()
            avg_loss = total_loss / len(eval_dataloader)
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("Total Loss: {:.1%}".format(total_loss))
            logger.info("Average Loss: {:.1%}".format(avg_loss))
            torch.cuda.empty_cache()


def do_inference(args,
                 model,
                 eval_dataloader,
                 criterion):
    device = "cuda"
    logger = logging.getLogger("simclr")
    logger.info("Enter inferencing")

    if device:
        model.to(device)

    model.eval()
    total_loss = 0

    for n_iter, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
    avg_loss = total_loss / len(eval_dataloader)
    logger.info("Validation Results ")
    logger.info("Total Loss: {:.1%}".format(total_loss))
    logger.info("Average Loss: {:.1%}".format(avg_loss))
    return avg_loss


