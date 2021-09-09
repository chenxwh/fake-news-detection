"""
Created on 02 Sep 2021
author: Chenxi
"""

import time
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import *


class KnowbertTrainer:
    def __init__(self, config, model, train_statements, train_labels, val_statements, val_labels):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': config['weight_decay']},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'])
        total_steps = (len(train_statements) // config['batch_size'] + 1) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                         num_training_steps=total_steps)
        self.model = model
        self.loss_fct = nn.CrossEntropyLoss()
        self.num_labels = config['num_labels']
        self.config = config
        self.train_statements = train_statements
        self.train_labels = train_labels
        self.val_statements = val_statements
        self.val_labels = val_labels
        self.model_save_path = os.path.join(config['save_path'],
                                            config['dataset_name'] + '-' + config['model_name'] + '-' + str(
                                                config['num_labels']) + '.pth')

    def train(self, verbose=False):
        if self.config['logging']:
            with open(self.config['logging_file'] + '.txt', 'a') as f:
                f.write(f"Dataset: {self.config['dataset_name']}, model: {self.config['model_name']}, batch size: {self.config['batch_size']}" + "\n")

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

            batches = self.config['batcher'].iter_batches(self.train_statements, self.train_labels)
            train_loop = 0
            for step, batch in enumerate(batches):
                train_loop += 1
                if verbose:
                    # Progress update every 50 batches.
                    if step % 50 == 0 and not step == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = format_time(time.time() - t0)
                        # Report progress.
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.
                              format(step, round(len(self.train_labels) // self.config['batch_size']), elapsed))

                tokens = to_device(batch['tokens'], self.config['device'])
                segment_ids = to_device(batch['segment_ids'], self.config['device'])
                candidates = to_device(batch['candidates'], self.config['device'])
                label_ids = to_device(batch['label_ids'], self.config['device'])

                self.model.zero_grad()
                result = self.model(tokens=tokens,
                                    segment_ids=segment_ids,
                                    candidates=candidates,
                                    label_ids=label_ids)
                loss = result.loss
                logits = result.logits

                total_train_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

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
            avg_train_loss = total_train_loss / train_loop
            ave_train_accuracy = total_train_accuracy / train_loop
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
            val_loop = 0
            for batch in self.config['batcher'].iter_batches(self.val_statements, self.val_labels):
                val_loop += 1
                tokens = to_device(batch['tokens'], self.config['device'])
                segment_ids = to_device(batch['segment_ids'], self.config['device'])
                candidates = to_device(batch['candidates'], self.config['device'])
                label_ids = to_device(batch['label_ids'], self.config['device'])

                with torch.no_grad():
                    result = self.model(tokens=tokens,
                                        segment_ids=segment_ids,
                                        candidates=candidates,
                                        label_ids=label_ids)
                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the softmax
                loss = result.loss
                logits = result.logits

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / val_loop
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
            avg_val_loss = total_eval_loss / len(self.val_labels)

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
