from pathlib import Path
from packaging import version
from datetime import datetime, timedelta
import logging

import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from param import parse_args
from pretrain_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict
from trainer_base import TrainerBase

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
        _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False
        return early_stop, get_better

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(args, train_loader, val_loader, test_loader, train)

        from pretrain_model import P5Pretraining
        model_kwargs = {}
        model_class = P5Pretraining

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(model_class, config, **model_kwargs)
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.model.tokenizer = self.tokenizer

        if args.load is not None:
            logging.info('Loading pretrained model')
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        if args.from_scratch:
            logging.info('Initializing weights from scratch')
            self.init_weights()

        logging.info(f'Model Launching at GPU {args.gpu}')
        self.model = self.model.to(args.gpu)

        super().__init__(args, train_loader, val_loader, test_loader, train)

        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
            if self.args.fp16:
                self.scaler = torch.cuda.amp.GradScaler()

        if args.multiGPU:
            self.model = DDP(self.model, device_ids=[args.gpu], find_unused_parameters=True)

    def train(self):
        early_stopping = EarlyStopping()
        global_step = 0

        for epoch in range(self.args.epoch):
            logging.info(f"Starting Epoch {epoch + 1}/{self.args.epoch}")
            self.model.train()
            epoch_loss = 0

            for step_i, batch in enumerate(self.train_loader):
                if self.args.fp16:
                    with torch.cuda.amp.autocast(enabled=self.args.fp16):
                        results = self.model.train_step(batch)
                    loss = results['loss']
                    self.scaler.scale(loss).backward()
                    
                    if self.args.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    results = self.model.train_step(batch)
                    loss = results['loss']
                    loss.backward()

                    if self.args.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                    self.optim.step()
                
                self.optim.zero_grad()
                epoch_loss += loss.item()
                global_step += 1

                if step_i % 100 == 0:
                    logging.info(f"Epoch {epoch + 1}, Step {step_i}, Loss {loss.item()}")

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logging.info(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss}")

            val_loss = self.evaluate_epoch(epoch)
            logging.info(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

            early_stop, get_better = early_stopping(val_loss)
            if get_better:
                self.save("BEST_EVAL_LOSS")
            self.save(f"Epoch{epoch + 1:02d}")

            if early_stop:
                logging.info("Early stopping triggered")
                break

        logging.info(f"Finished Epoch {epoch + 1}/{self.args.epoch}")


    def evaluate_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        loss_meter = LossMeter()

        with torch.no_grad():
            logging.info(f"Starting evaluation for Epoch {epoch + 1}")
            pbar = tqdm(total=len(self.val_loader), desc=f"Evaluating Epoch {epoch + 1}")

            for step_i, batch in enumerate(self.val_loader):
                try:
                    with torch.cuda.amp.autocast(enabled=self.args.fp16):
                        results = self.model.valid_step(batch)
                    loss = results['loss']
                    val_loss += loss.item()
                    loss_meter.update(loss.item())

                    if step_i % 100 == 0:
                        logging.info(f"Validation Step {step_i}/{len(self.val_loader)}, Loss: {loss.item():.4f}, Running Average Loss: {loss_meter.val:.4f}")

                    pbar.update(1)
                except KeyError as e:
                    logging.error(f"KeyError during validation at step {step_i}: {e}")
                    continue

            pbar.close()

        avg_val_loss = val_loss / len(self.val_loader)
        logging.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss
    

def initialize_distributed_training(args):
    if args.distributed:
        logging.info(f"Initializing distributed training on GPU {args.gpu}")
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=6000))

def build_data_loader(args, gpu, mode, task_list, sample_numbers):
    logging.info(f"----Building {mode} loader at GPU {gpu}----")
    loader = get_loader(
        args,
        task_list,
        sample_numbers,
        split=getattr(args, mode),
        mode=mode,
        batch_size=args.batch_size if mode == 'train' else args.val_batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )
    return loader

def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    logging.info(f"Process Launching at GPU {gpu}")

    initialize_distributed_training(args)

    train_task_list = {'sequential': ['1-1']}
    train_sample_numbers = {'sequential': 1}
    train_loader = build_data_loader(args, gpu, 'train', train_task_list, train_sample_numbers)

    val_task_list = {'sequential': ['1-1']}
    val_sample_numbers = {'sequential': 1}
    val_loader = build_data_loader(args, gpu, 'valid', val_task_list, val_sample_numbers)

    trainer = Trainer(args, train_loader, val_loader, train=True)
    trainer.train()

if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.local_rank in [0, -1]:
        logging.info(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        logging.info(LOSSES_NAME)

    LOSSES_NAME.extend(['pair_loss', 'total_loss'])
    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = ['MIND'] if 'MIND' in args.train else []
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    current_time = datetime.now().strftime('%b%d_%H-%M')
    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if comments:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
    else:
        main_worker(0, args)  # For single GPU setup


    
    
    # def train(self):
    #     early_stopping = EarlyStopping()
    #     LOSSES_NAME = self.args.LOSSES_NAME

    #     if self.args.dry:
    #         results = self.evaluate_epoch(epoch=0)

    #     if self.verbose:
    #         loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
    #         best_eval_loss = 100000.

    #         if 't5' in self.args.backbone:
    #             project_name = "T5_Pretrain"

    #         src_dir = Path(__file__).resolve().parent
    #         base_path = str(src_dir.parent)
    #         src_dir = str(src_dir)

    #     if self.args.distributed:
    #         dist.barrier()

    #     global_step = 0
    #     for epoch in range(self.args.epoch):

    #         if self.start_epoch is not None:
    #             epoch += self.start_epoch

    #         if self.args.distributed:
    #             self.train_loader.sampler.set_epoch(epoch)

    #         # Train
    #         self.model.train()

    #         if self.verbose:
    #             pbar = tqdm(total = len(self.train_loader), ncols=275)

    #         epoch_results = {}
    #         for loss_name in LOSSES_NAME:
    #             epoch_results[loss_name] = 0.
    #             epoch_results[f'{loss_name}_count'] = 0


    #         for step_i, batch in enumerate(self.train_loader):

    #             if self.args.fp16 and _use_native_amp:
    #                 with autocast():
    #                     if self.args.distributed:
    #                         results = self.model.module.train_step(batch)
    #                     else:
    #                         results = self.model.train_step(batch)
    #             else:
    #                 if self.args.distributed:
    #                     results = self.model.module.train_step(batch)
    #                 else:
    #                     results = self.model.train_step(batch)

    #             loss = results['loss']

    #             if self.args.fp16 and _use_native_amp:
    #                 self.scaler.scale(loss).backward()
    #             elif self.args.fp16 and _use_apex:
    #                 with amp.scale_loss(loss, self.optim) as scaled_loss:
    #                     scaled_loss.backward()
    #             else:
    #                 loss.backward()

    #             loss = loss.detach()


    #             # Update Parameters
    #             if self.args.clip_grad_norm > 0:
    #                 if self.args.fp16 and _use_native_amp:
    #                     self.scaler.unscale_(self.optim)
    #                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
    #                 elif self.args.fp16 and _use_apex:
    #                     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
    #                 else:
    #                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

    #             if self.args.fp16 and _use_native_amp:
    #                 self.scaler.step(self.optim)
    #                 self.scaler.update()
    #             else:
    #                 self.optim.step()

    #             if self.lr_scheduler:
    #                 self.lr_scheduler.step()

    #             # self.model.zero_grad()
    #             for param in self.model.parameters():
    #                 param.grad = None

    #             global_step += 1

    #             if self.lr_scheduler:
    #                 if version.parse(torch.__version__) >= version.parse("1.4"):
    #                     lr = self.lr_scheduler.get_last_lr()[0]
    #                 else:
    #                     lr = self.lr_scheduler.get_lr()[0]
    #             else:
    #                 try:
    #                     lr = self.optim.get_lr()[0]
    #                 except AttributeError:
    #                     lr = self.args.lr


    #             # update sequential loss and sequential loss count
    #             for k, v in results.items():
    #                 if k in epoch_results:
    #                     if isinstance(v, int):
    #                         epoch_results[k] += v
    #                     elif isinstance(v, torch.Tensor):
    #                         epoch_results[k] += v.item()


    #             if self.verbose :
    #                 desc_str = f'Step_i {step_i} | Epoch {epoch} | LR {lr:.6f} |'

    #                 for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
    #                     if loss_name in results:
    #                         loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
    #                     if len(loss_meter) > 0:
    #                         loss_count = epoch_results[f'{loss_name}_count']
    #                         desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

    #                 if step_i % 800 == 0:
    #                     pbar.set_description(desc_str)
    #                     pbar.update(800)

    #         if self.verbose:
    #             pbar.close()

    #         dist.barrier()

    #         results = reduce_dict(epoch_results, average=False)
    #         # for each epoch
    #         if self.verbose:
    #             train_loss = results['total_loss']
    #             train_loss_count = results['total_loss_count']
    #             avg_train_loss = train_loss / train_loss_count

    #             losses_str = f"Total train Loss: {train_loss:.3f}"
    #             losses_str += f"Average train Loss: {avg_train_loss:.3f}\n"

    #             for name, loss in results.items():
    #                 if name[-4:] == 'loss':
    #                     loss_count = int(results[name + '_count'])
    #                     if loss_count > 0:
    #                         avg_loss = loss / loss_count
    #                         losses_str  += f"{name} ({loss_count}): {avg_loss:.3f} "
    #             print(losses_str)

    #         dist.barrier()



    #         if epoch % 1 == 0:
    #             valid_results = self.evaluate_epoch(epoch=epoch)
    #             valid_results = reduce_dict(valid_results, average=False)

    #             if self.verbose:
    #                 valid_loss = valid_results['total_loss']
    #                 valid_loss_count = valid_results['total_loss_count']

    #                 avg_valid_loss = valid_loss / valid_loss_count
    #                 losses_str = f"Total Valid Loss: {valid_loss:.3f}"
    #                 losses_str += f"Valid Loss: {avg_valid_loss:.3f}\n"

    #                 for name, loss in valid_results.items():
    #                     if name[-4:] == 'loss':
    #                         loss_count = int(valid_results[name + '_count'])
    #                         if loss_count > 0:
    #                             avg_loss = loss / loss_count
    #                             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

    #                 losses_str += '\n'
    #                 print(losses_str)

    #             dist.barrier()

    #             if self.verbose:
    #                 # Save
    #                 if avg_valid_loss < best_eval_loss:
    #                     best_eval_loss = avg_valid_loss
    #                     self.save("BEST_EVAL_LOSS")
    #                 self.save("Epoch%02d" % (epoch + 1))

    #                 early_stop, _ = early_stopping(avg_valid_loss)
    #                 if early_stop:
    #                     print('Early Stop')
    #                     break

    #             dist.barrier()


    

    # def evaluate_epoch(self, epoch):

    #     LOSSES_NAME = self.args.LOSSES_NAME 
    #     epoch_results = {}
    #     for loss_name in LOSSES_NAME:
    #         epoch_results[loss_name] = 0.
    #         epoch_results[f'{loss_name}_count'] = 0


    #     self.model.eval()
    #     with torch.no_grad():
    #         if self.verbose:
    #             loss_meter = LossMeter()
    #             loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

    #             pbar = tqdm(total=len(self.val_loader), ncols=275)

    #         for step_i, batch in enumerate(self.val_loader):

    #             if self.args.distributed:
    #                 results = self.model.module.valid_step(batch)
    #             else:
    #                 results = self.model.module.valid_step(batch)


    #             for k, v in results.items():
    #                 if k in epoch_results:
    #                     if isinstance(v, int):
    #                         epoch_results[k] += v
    #                     elif isinstance(v, torch.Tensor):
    #                         epoch_results[k] += v.item()

    #             if self.verbose:
    #                 desc_str = f'Valid Epoch {epoch} |'
    #                 for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

    #                     if loss_name in results:
    #                         # append a batch_loss
    #                         loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
    #                     if len(loss_meter) > 0:
    #                         loss_count = epoch_results[f'{loss_name}_count']
    #                         desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}' # average over batches

    #                 if step_i % 50 == 0:
    #                     pbar.set_description(desc_str)
    #                     pbar.update(50)
    #             dist.barrier()

    #         if self.verbose:
    #             pbar.close()
    #         dist.barrier()
    #         return epoch_results

# def main_worker(gpu, args):
#     args.gpu = gpu
#     args.rank = gpu
#     print(f'Process Launching at GPU {gpu}')

#     if args.distributed:
#         import datetime
#         torch.cuda.set_device(args.gpu)
#         dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=6000))

#     print(f'----Building train loader at GPU {gpu}----')

#     train_task_list = {'sequential':['1-1']}
#     train_sample_numbers = {'sequential': 1}
#     train_loader = get_loader(
#         args,
#         train_task_list,
#         train_sample_numbers,
#         split = args.train,
#         mode = 'train',
#         batch_size = args.batch_size,
#         workers = args.num_workers,
#         distributed = args.distributed
#     )

#     print(f'----Building val loader at GPU {gpu}----')

#     val_task_list = {'sequential':['1-1']}
#     val_sample_numbers = { 'sequential': 1}
#     val_loader = get_loader(
#         args,
#         val_task_list,
#         val_sample_numbers,
#         split = args.valid,
#         mode = 'val',
#         batch_size = args.val_batch_size,
#         workers = args.num_workers,
#         distributed = args.distributed
#     )

#     trainer = Trainer(args, train_loader, val_loader, train=True)
#     trainer.train()



# if __name__ == "__main__":
#     cudnn.benchmark = True
#     args = parse_args()
#     if args.local_rank in [0, -1]:
#         print(args)

#     ngpus_per_node = torch.cuda.device_count()
#     args.world_size = ngpus_per_node

#     LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
#     if args.local_rank in [0, -1]:
#         print(LOSSES_NAME) 

#     LOSSES_NAME.append('pair_loss')
#     LOSSES_NAME.append('total_loss')

#     args.LOSSES_NAME = LOSSES_NAME

#     comments = []
#     dsets = []
#     if 'MIND' in args.train:
#         dsets.append('MIND')

#     comments.append(''.join(dsets))

#     if args.backbone:
#         comments.append(args.backbone)
#     comments.append(''.join(args.losses.split(',')))

#     if args.comment != '':
#         comments.append(args.comment)
#     comment = '_'.join(comments)

#     from datetime import datetime
#     current_time = datetime.now().strftime('%b%d_%H-%M')

#     project_dir = Path(__file__).resolve().parent.parent

#     if args.local_rank in [0, -1]:
#         run_name = f'{current_time}_GPU{args.world_size}'
#         if len(comments) > 0:
#             run_name += f'_{comment}'
#         args.run_name = run_name

#     if args.distributed:
#         main_worker(args.local_rank, args)