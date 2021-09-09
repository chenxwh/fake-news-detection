"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import time
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import *


class Trainer:
    def __init__(self, config, model, train_dataloader, val_dataloader):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': config['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'])
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=len(train_dataloader) * config['epochs'])
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()
        self.num_labels = config['num_labels']
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if config['model_name'] == 'ernie':
            self.model_save_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name']
                                                + '-' + str(config['num_labels']) + '-thr-' + str(config['tagme_thr']) +'.pth')
        else:
            self.model_save_path = os.path.join(config['save_path'], config['dataset_name'] + '-' + config['model_name']
                                            + '-' + str(config['num_labels']) + '.pth')

    def train(self, verbose=False):
        if self.config['logging']:
            with open(self.config['logging_file'] + '.txt', 'a') as f:
                f.write(f"Dataset: {self.config['dataset_name']}, model: {self.config['model_name']},"
                        f" batch size: {self.config['batch_size']}" + "\n")
                if self.config['model_name'] == 'ernie':
                    f.write(f"threshold: {self.config['tagme_thr']}" + "\n")

        best_accuracy = 0

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        for epoch_i in range(0, self.config['epochs']):

            # ========================================
            #               Training
            # ========================================

            print("")
            print(f'======== Epoch {epoch_i + 1} / {self.config["epochs"]} ========')
            print('Training...')

            if self.config['logging']:
                with open(self.config['logging_file'] + '.txt', 'a') as f:
                    f.write("" + "\n")
                    f.write(f'======== Epoch {epoch_i + 1} / {self.config["epochs"]} ========' + "\n")
                    f.write('Training...' + "\n")

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0
            total_train_accuracy = 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                if verbose:
                    # Progress update every 50 batches.
                    if step % 50 == 0 and not step == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = format_time(time.time() - t0)
                        # Report progress.
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_dataloader),
                                                                                    elapsed))

                self.model.zero_grad()

                if self.config['model_name'] == 'ernie':
                    embed = self.config['ernie_embed']
                    b_input_ids = batch['input_ids'].to(self.config['device'])
                    b_input_ent = embed(batch['input_ent'] + 1).to(self.config['device'])
                    b_ent_mask = batch['ent_mask'].to(self.config['device'])
                    b_labels = batch['labels'].to(self.config['device'])

                    result = self.model(b_input_ids,
                                        input_ent=b_input_ent,
                                        ent_mask=b_ent_mask,
                                        labels=b_labels)
                    loss = result.loss
                    logits = result.logits

                else:
                    b_input_ids = batch['input_ids'].to(self.config['device'])
                    b_input_mask = batch['attention_mask'].to(self.config['device'])
                    b_labels = batch['labels'].to(self.config['device'])

                    if self.config['model_name'] in ['k_adapter']:
                        logits = self.model(b_input_ids,
                                            attention_mask=b_input_mask)
                        loss = self.loss_fct(logits.view(-1, self.num_labels), b_labels.view(-1))
                    else:

                        result = self.model(b_input_ids,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            return_dict=True)

                        loss = result.loss
                        logits = result.logits

                total_train_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_train_accuracy += flat_accuracy(logits, label_ids)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                self.optimizer.step()

                # Update the learning rate.
                self.scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            ave_train_accuracy = total_train_accuracy / len(self.train_dataloader)
            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training Accuracy: {0:.2f}".format(ave_train_accuracy))
            print(f'  Training epoch took: {training_time}')

            if self.config['logging']:
                with open(self.config['logging_file'] + '.txt', 'a') as f:
                    f.write("" + "\n")
                    f.write("  Average training loss: {0:.2f}".format(avg_train_loss) + "\n")
                    f.write("  Training Accuracy: {0:.2f}".format(ave_train_accuracy) + "\n")
                    f.write(f'  Training epoch took: {training_time}' + "\n")

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            if self.config['logging']:
                with open(self.config['logging_file'] + '.txt', 'a') as f:
                    f.write("" + "\n")
                    f.write("Running Validation..." + "\n")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in self.val_dataloader:

                with torch.no_grad():

                    if self.config['model_name'] == 'ernie':

                        embed = self.config['ernie_embed']
                        b_input_ids = batch['input_ids'].to(self.config['device'])
                        b_input_ent = embed(batch['input_ent'] + 1).to(self.config['device'])
                        b_ent_mask = batch['ent_mask'].to(self.config['device'])
                        b_labels = batch['labels'].to(self.config['device'])

                        result = self.model(b_input_ids,
                                            input_ent=b_input_ent,
                                            ent_mask=b_ent_mask,
                                            labels=b_labels)

                        loss = result.loss
                        logits = result.logits

                    else:
                        b_input_ids = batch['input_ids'].to(self.config['device'])
                        b_input_mask = batch['attention_mask'].to(self.config['device'])
                        b_labels = batch['labels'].to(self.config['device'])

                        if self.config['model_name'] in ['k_adapter']:
                            logits = self.model(b_input_ids,
                                                attention_mask=b_input_mask)
                            loss = self.loss_fct(logits.view(-1, self.num_labels), b_labels.view(-1))
                        else:

                            result = self.model(b_input_ids,
                                                attention_mask=b_input_mask,
                                                labels=b_labels,
                                                return_dict=True)

                            loss = result.loss
                            logits = result.logits

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.val_dataloader)
            print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
            if self.config['logging']:
                with open(self.config['logging_file'] + '.txt', 'a') as f:
                    f.write("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy) + "\n")

            # save the best model
            if avg_val_accuracy > best_accuracy:
                os.makedirs(self.config['save_path'], exist_ok=True)
                torch.save(self.model.state_dict(), self.model_save_path)
                best_accuracy = avg_val_accuracy

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(self.val_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            if self.config['logging']:
                with open(self.config['logging_file'] + '.txt', 'a') as f:
                    f.write("  Validation Loss: {0:.2f}".format(avg_val_loss) + "\n")
                    f.write("  Validation took: {:}".format(validation_time) + "\n")

        print("")
        print("Training complete!")
        print(f'Best Validation Accuracy: {best_accuracy}')
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        if self.config['logging']:
            with open(self.config['logging_file'] + '.txt', 'a') as f:
                f.write("" + "\n")
                f.write("Training complete!" + "\n")
                f.write(f'Best Validation Accuracy: {best_accuracy}' + "\n")
                f.write("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)) + "\n")

