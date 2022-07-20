import datetime
import random
import time

import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm


from custom_dataset import CustomDataset
from custom_model import CustomModel


random.seed(24785)
torch.manual_seed(24785)

BATCH_SIZE = 32
NB_EPOCHS = 10

tensorboard_writer = SummaryWriter(writer_dir)

# %%

TRAIN_DATA_FOLDER = "XXX"
VALID_DATA_FOLDER = "XXX"
date = datetime.datetime.now()
tmp_name = 'leo_explo_' + datetime.datetime.strftime(date, '%H%M')


train_dataset = CustomDataset(TRAIN_DATA_FOLDER)
valid_dataset = CustomDataset(VALID_DATA_FOLDER)

# TODO: compute samples_weights
sampler = WeightedRandomSampler(
    weights=samples_weights,
    num_samples=len(samples_weights),
    replacement=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler)

valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler)


# %%

model = CustomModel()
optimizer = torch.optim.Adam(model.parameters())

# %%

# TODO: compute class weights
criterion = BCEWithLogitsLoss(weight=class_weight, reduction='none')


for epoch in range(NB_EPOCHS):
    print(f'Epoch {epoch}:')
    epoch_train_losses = []
    epoch_valid_losses = []
    model.train()

    stop = time.time()
    for i, (input, target) in enumerate(tqdm(train_dataloader)):
        if i < 1:
            tensorboard_writer.add_figure('test', input[0])

        if input is None:
            continue
        start = time.time()

        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.detach().cpu())
        # self.trackershub.record(
        #     'Training iteration loss',
        #     epoch_train_losses[-1])

        stop = time.time()
        # self.trackershub.record('Training inference', stop-start)

        # proba = nn.Sigmoid()(output).cpu().detach().numpy()
        # try:
        #     preds = list(np.where(proba > self.threshold, 1, 0))
        #     metrics = compute_metrics(preds, target)
        #     for metric_name in ['TP', 'FP', 'TN', 'FN']:
        #         self.trackershub.record(f'Train {metric_name} aux',
        #                                 metrics[metric_name],
        #                                 ['intra_epoch'],
        #                                 tensorboard_display=False)
        # except Exception as e:
        #     logging.warn(
        #         f'Exception raised when updating metrics: {e}')

        # if i == 0:
        #     self.trackershub.write_batch(
        #         'Train batch', input, target, proba, None, names,
        #         shown_batch_size=8)
        # self.trackershub.record_scores(
        #     'Train_probas', target, proba)
        # self.trackershub.record_scores(
        #     'Train_scores', target, output.detach().numpy())
        # self.trackershub.record_scores(
        #     'Train_losses', target, loss_per_sample.detach().numpy())

        # for scheduler in schedulers:
        #     scheduler.step()
        # Sanity check, should always be true with current parameters
        # assert len(scheduler.get_last_lr())
        # self.trackershub.record('Learning Rate', scheduler.get_lr()[0])

    # self.trackershub.write_recorded_scores('Train_probas', epoch)
    # self.trackershub.write_recorded_scores('Train_scores', epoch,
    #                                        range=None)
    # self.trackershub.write_recorded_scores('Train_losses', epoch,
    #                                        range=None)

    torch.save(model.state_dict(), tmp_name)
    model.eval()
    for i, (input, target) in enumerate(tqdm(valid_dataloader)):
        # if i > 1:
        #     break
        if input is None:
            continue
        start = time.time()
        # if i > 0:
        #     self.trackershub.record('valid data loading', start-stop)
        output = model(input).view(-1)

        loss_per_sample = criterion(output, target.float())
        loss = loss_per_sample.mean()

        # proba = nn.Sigmoid()(output).cpu().detach().numpy()
        # proba = list(proba[:])

        epoch_valid_losses.append(loss.detach().cpu())
        # self.trackershub.record(
        #     'valid iteration loss',
        #     epoch_valid_losses[-1])

        # try:
        #     preds = list(np.where(proba > self.threshold, 1, 0))
        #     metrics = compute_metrics(preds, target)
        #     for metric_name in ['TP', 'FP', 'TN', 'FN']:
        #         self.trackershub.record(f'valid {metric_name} aux',
        #                                 metrics[metric_name],
        #                                 ['intra_epoch'],
        #                                 tensorboard_display=False)
        # except Exception as e:
        #     logging.warn(
        #         f'Exception raised when updating metrics: {e}')

        stop = time.time()
        # self.trackershub.record_scores(
        #     'valid_proba', target, proba)
        # self.trackershub.record_scores(
        #     'valid_scores', target, output.detach().numpy())
        # self.trackershub.record_scores(
        #     'valid_losses', target, loss_per_sample.detach().numpy())
        # self.trackershub.record('valid inference', stop-start)

        # if i == 0:
        #     proba = nn.Sigmoid()(output).cpu().detach().numpy()
        #     self.trackershub.write_batch(
        #         'valid batch', input, target, proba, None, names,
        #         shown_batch_size=8)

    # self.trackershub.write_recorded_scores('valid_proba', epoch)
    # self.trackershub.write_recorded_scores('valid_scores', epoch,
    #                                        range=None)
    # self.trackershub.write_recorded_scores('valid_losses', epoch,
    #                                        range=None)

    # if self.write_to_blob:
    #     self.weights_connector.upload_torch_weights(
    #         self,
    #         self.xp_name + '_weights.pt',
    #         upload_folder_name=REMOTE_WEIGHTS_PATH,
    #         local_folder_name=LOCAL_WEIGHT_TMP_FOLDER,
    #         force=True)

    train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)
    # self.trackershub.record('valid epoch loss', valid_loss)
    tensorboard_writer.add_scalar(
        'Training epoch loss',
        train_loss,
        epoch)
    tensorboard_writer.add_scalar(
        'Valid epoch loss',
        valid_loss,
        epoch)

    # for mode in ['valid', 'Train']:
    #     for metric_name in ['TP', 'FP', 'TN', 'FN']:
    #         metric_value = self.trackershub.get_tracker_sum(
    #             f'{mode} {metric_name} aux')
    #         self.trackershub.record(
    #             f'{mode} {metric_name}',
    #             metric_value)
    #     for name, val1_name, val2_name in [
    #             ('Precision', 'TN', 'FN'), ('Recall', 'TN', 'FP')]:
    #         val1 = self.trackershub.get_tracker_sum(
    #             f'{mode} {val1_name} aux')
    #         val2 = self.trackershub.get_tracker_sum(
    #             f'{mode} {val2_name} aux')
    #         self.trackershub.record(
    #             f'{mode} {name}',
    #             val1 / (val1 + val2))

    print(f'train_loss: {train_loss}')
    print(f'valid_loss: {valid_loss}')

# self.trackershub.print_averages('time')
# self.trackershub.print_totals('time')
# self.trackershub.plot_evolutions('loss')