"""
Created on 02 Sep 2021
author: Chenxi
"""

import os
import time
import torch
import torch.nn as nn
from utils import *
from trainer import Trainer


class KnowbertTrainer(Trainer):

    def train(self, verbose=False):
        best_accuracy = 0
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        for epoch_i in range(0, self.config['epochs']):

            # ========================================
            #               Training
            # ========================================

            print("")
            print(f'======== Epoch {epoch_i + 1} / {self.config["epochs"]} ========')
            print('Training...')

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

                b_input_ids = batch['input_ids'].to(self.config['device'])
                b_candidates = batch['candidates'].to(self.config['device'])
                b_labels = batch['labels'].to(self.config['device'])

                self.model.zero_grad()

                result = self.model(tokens=b_input_ids,
                                    candidates=b_candidates,
                                    labels=b_labels)

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
            print(f'  Training epcoh took: {training_time}')

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in self.val_dataloader:

                b_input_ids = batch['input_ids'].to(self.config['device'])
                b_candidates = batch['candidates'].to(self.config['device'])
                b_labels = batch['labels'].to(self.config['device'])

                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    result = self.model(tokens=b_input_ids,
                                        candidates=b_candidates,
                                        labels=b_labels)

                # Get the loss and "logits" output by the model. The "logits" are the
                # output values prior to applying an activation function like the softmax
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

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")
        print(f'Best Validation Accuracy: {best_accuracy}')
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
