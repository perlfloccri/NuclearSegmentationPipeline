
from __future__ import print_function

import os
import sys
import time
import pickle
import itertools
import numpy as np

import theano
import lasagne
from lasagne.utils import floatX

from lasagne_wrapper.utils import BColors, print_net_architecture
import theano.tensor as T

from lasagne_wrapper.data_pool import DataPool
from lasagne_wrapper.batch_iterators import threaded_generator_from_iterator


class BaseNetwork_(object):
    """
    Neural Network
    """

    def __init__(self, net, print_architecture=True):
        """
        Constructor
        """
        self.net = net
        self.compute_output = None
        self.compute_output_dict = dict()

        # get input shape of network
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        self.input_shape = l_in.output_shape

        if print_architecture:
            print_net_architecture(net)

    def fit(self, data, training_strategy, dump_file=None, log_file=None):
        """ Train model """
        print("Training neural network...")
        col = BColors()

        # create data pool if raw data is given
        if "X_train" in data:
            data_pools = dict()
            data_pools['train'] = DataPool(data['X_train'], data['y_train'])
            data_pools['valid'] = DataPool(data['X_valid'], data['y_valid'])
        else:
            data_pools = data

        # check if out_path exists
        if dump_file is not None:
            out_path = os.path.dirname(dump_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # log model evolution
        if log_file is not None:
            out_path = os.path.dirname(log_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # adaptive learning rate
        learn_rate = training_strategy.ini_learning_rate
        learning_rate = theano.shared(floatX(learn_rate))
        learning_rate.set_value(training_strategy.adapt_learn_rate(training_strategy.ini_learning_rate, 0))

        # initialize evaluation output
        pred_tr_err, pred_val_err, overfitting = [], [], []
        tr_accs, va_accs = [], []

        print("Compiling theano train functions...")
        iter_funcs = self._create_iter_functions(y_tensor_type=training_strategy.y_tensor_type,
                                                 objective=training_strategy.objective, learning_rate=learning_rate,
                                                 l_2=training_strategy.L2,
                                                 compute_updates=training_strategy.update_parameters,
                                                 use_weights=training_strategy.use_weights)

        print("Starting training...")
        now = time.time()
        try:

            # initialize early stopping
            last_improvement = 0
            best_model = lasagne.layers.get_all_param_values(self.net)

            # iterate training epochs
            prev_tr_loss, prev_va_loss = np.inf, np.inf
            for epoch in self._train(iter_funcs, data_pools, training_strategy.build_train_batch_iterator(),
                                     training_strategy.build_valid_batch_iterator()):

                print("Epoch {} of {} took {:.3f}s".format(epoch['number'], training_strategy.max_epochs,
                                                           time.time() - now))
                now = time.time()

                # --- collect train output ---

                tr_loss, va_loss = epoch['train_loss'], epoch['valid_loss']
                overfit = epoch['overfitting']

                # prepare early stopping
                improvement = va_loss < prev_va_loss
                if improvement:
                    last_improvement = 0
                    best_model = lasagne.layers.get_all_param_values(self.net)
                    best_epoch = epoch['number']

                    # dump net parameters during training
                    if dump_file is not None:
                        with open(dump_file, 'wb') as fp:
                            pickle.dump(best_model, fp)

                last_improvement += 1

                # print train output
                txt_tr = 'costs_tr %.5f ' % tr_loss
                if tr_loss < prev_tr_loss:
                    txt_tr = col.print_colored(txt_tr, BColors.OKGREEN)
                    prev_tr_loss = tr_loss

                txt_val = 'costs_val %.5f ' % va_loss
                if va_loss < prev_va_loss:
                    txt_val = col.print_colored(txt_val, BColors.OKGREEN)
                    prev_va_loss = va_loss

                print('  lr: %.5f' % learn_rate)
                print('  ' + txt_tr + txt_val + 'tr/val %.3f' % overfit)

                # collect model evolution data
                pred_tr_err.append(tr_loss)
                pred_val_err.append(va_loss)
                overfitting.append(overfit)

                # save results
                exp_res = dict()
                exp_res['pred_tr_err'] = pred_tr_err
                exp_res['pred_val_err'] = pred_val_err
                exp_res['overfitting'] = overfitting

                if log_file is not None:
                    with open(log_file, 'w') as fp:
                        pickle.dump(exp_res, fp)

                # --- early stopping: preserve best model ---
                if last_improvement > training_strategy.patience:
                    print(col.print_colored("Early Stopping!", BColors.WARNING))
                    status = "Epoch: %d, Best Validation Loss: %.5f" % (
                        best_epoch, prev_va_loss)
                    print(col.print_colored(status, BColors.WARNING))
                    break

                # maximum number of epochs reached
                if epoch['number'] >= training_strategy.max_epochs:
                    break

                # update learning rate
                learn_rate = training_strategy.adapt_learn_rate(learn_rate, epoch['number'])
                learning_rate.set_value(learn_rate)

        except KeyboardInterrupt:
            pass

        # set net to best weights
        lasagne.layers.set_all_param_values(self.net, best_model)

    def predict_proba(self, input):
        """
        Predict on test samples
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()

        return self.compute_output(*input)

    def predict(self, input):
        """
        Predict class labels on test samples
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        return np.argmax(self.predict_proba(*input), axis=-1)

    def compute_layer_output(self, input, layer):
        """
        Compute output of given layer
        layer: either a string (name of layer) or a layer object
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        # get layer by name
        if not isinstance(layer, lasagne.layers.Layer):
            for l in lasagne.layers.helper.get_all_layers(self.net):
                if l.name == layer:
                    layer = l
                    break

        # compile prediction function for target layer
        if layer not in self.compute_output_dict:
            self.compute_output_dict[layer] = self._compile_prediction_function(target_layer=layer)

        return self.compute_output_dict[layer](*input)

    def save(self, file_path):
        """
        Save model to disk
        """
        with open(file_path, 'w') as fp:
            params = lasagne.layers.get_all_param_values(self.net)
            pickle.dump(params, fp, -1)

    def load(self, file_path):
        """
        load model from disk
        """
        with open(file_path, 'r') as fp:
            params = pickle.load(fp)
        lasagne.layers.set_all_param_values(self.net, params)

    def _compile_prediction_function(self, target_layer=None):
        """
        Compile theano prediction function
        """

        # collect input vars
        all_layers = lasagne.layers.helper.get_all_layers(self.net)
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        # get network output nad compile function
        if target_layer is None:
            target_layer = self.net

        net_output = lasagne.layers.get_output(target_layer, deterministic=True)
        return theano.function(inputs=input_vars, outputs=net_output)

    def _create_iter_functions(self, y_tensor_type, objective, learning_rate, l_2, compute_updates, use_weights):
        """ Create functions for training, validation and testing to iterate one epoch. """

        # init target tensor
        targets = y_tensor_type('y')
        weights = y_tensor_type('w')

        # get input layer
        all_layers = lasagne.layers.helper.get_all_layers(self.net)

        # collect input vars
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        # compute train costs
        tr_output = lasagne.layers.get_output(self.net, deterministic=False)

        if use_weights:
            tr_cost = objective(tr_output, targets, weights)
            tr_input = input_vars + [targets, weights]
        else:
            tr_cost = objective(tr_output, targets)
            tr_input = input_vars + [targets]

        # regularizer for RNNs
        for l in all_layers:

            if l.name == "norm_reg_rnn":

                H = lasagne.layers.get_output(l, deterministic=False)
                H_l2 = T.sqrt(T.sum(H**2, axis=-1))
                norm_diffs = (H_l2[:, 1:] - H_l2[:, :-1])**2
                norm_preserving_loss = T.mean(norm_diffs)

                beta = 1.0
                tr_cost += beta * norm_preserving_loss

            else:
                pass

        # compute validation costs
        va_output = lasagne.layers.get_output(self.net, deterministic=True)
        va_cost = objective(va_output, targets)

        # collect all parameters of net and compute updates
        all_params = lasagne.layers.get_all_params(self.net, trainable=True)

        # add weight decay
        if l_2 is not None:
            all_layers = lasagne.layers.get_all_layers(self.net)
            tr_cost += l_2 * lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2)

        # compute updates
        all_grads = lasagne.updates.get_or_compute_grads(tr_cost, all_params)
        updates = compute_updates(all_grads, all_params, learning_rate)

        # compile iter functions
        tr_outputs = [tr_cost, tr_output]
        iter_train = theano.function(tr_input, tr_outputs, updates=updates)

        va_inputs = input_vars + [targets]
        va_outputs = [va_cost, va_output]
        iter_valid = theano.function(va_inputs, va_outputs)

        return dict(train=iter_train, valid=iter_valid, test=iter_valid)

    def _train(self, iter_funcs, data_pools, train_batch_iter, valid_batch_iter):
        """
        Train the model with `dataset` with mini-batch training.
        Each mini-batch has `batch_size` recordings.
        """
        col = BColors()

        for epoch in itertools.count(1):

            # iterate train batches
            batch_train_losses = []
            iterator = train_batch_iter(data_pools['train'])
            generator = threaded_generator_from_iterator(iterator)

            batch_times = np.zeros(5, dtype=np.float32)
            start, after = time.time(), time.time()
            for i_batch, train_input in enumerate(generator):
                batch_res = iter_funcs['train'](*train_input)
                batch_train_losses.append(batch_res[0])

                # train time
                batch_time = time.time() - after
                after = time.time()
                train_time = (after - start)

                # estimate updates per second (running avg)
                batch_times[0:4] = batch_times[1:5]
                batch_times[4] = batch_time
                ups = 1.0 / batch_times.mean()

                # report loss during training
                perc = 100 * (float(i_batch) / train_batch_iter.n_batches)
                dec = int(perc // 4)
                progbar = "|" + dec * "#" + (25 - dec) * "-" + "|"
                vals = (perc, progbar, train_time, ups, np.mean(batch_train_losses))
                loss_str = " (%d%%) %s time: %.2fs, ups: %.2f, loss: %.5f" % vals
                print(col.print_colored(loss_str, col.WARNING), end="\r")
                sys.stdout.flush()

            # print("\x1b[K", end="\r")
            print(' ')
            print(' ')
            avg_train_loss = np.mean(batch_train_losses)

            # evaluate classification power of data set

            # iterate validation batches
            batch_valid_losses = []
            iterator = valid_batch_iter(data_pools['valid'])
            generator = threaded_generator_from_iterator(iterator)

            for va_input in generator:
                batch_res = iter_funcs['valid'](*va_input)
                batch_valid_losses.append(batch_res[0])

            avg_valid_loss = np.mean(batch_valid_losses)

            # collect results
            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
                'overfitting': avg_train_loss / avg_valid_loss,
            }


class Network(object):
    """
    Neural Network
    """

    def __init__(self, net, print_architecture=True):
        """
        Constructor
        """
        self.net = net
        self.compute_output = None
        self.compute_output_dict = dict()

        # get input shape of network
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        self.input_shape = l_in.output_shape
        
        if print_architecture:
            print_net_architecture(net)

    def fit(self, data, training_strategy, dump_file=None, log_file=None):
        """ Train model """
        print("Training neural network...")
        col = BColors()

        # create data pool if raw data is given
        if "X_train" in data:
            data_pools = dict()
            data_pools['train'] = DataPool(data['X_train'], data['y_train'])
            data_pools['valid'] = DataPool(data['X_valid'], data['y_valid'])
        else:
            data_pools = data

        # check if out_path exists
        if dump_file is not None:
            out_path = os.path.dirname(dump_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # log model evolution
        if log_file is not None:
            out_path = os.path.dirname(log_file)
            if out_path != '' and not os.path.exists(out_path):
                os.mkdir(out_path)

        # adaptive learning rate
        learn_rate = training_strategy.ini_learning_rate
        learning_rate = theano.shared(floatX(learn_rate))
        learning_rate.set_value(training_strategy.adapt_learn_rate(training_strategy.ini_learning_rate, 0))

        # initialize evaluation output
        pred_tr_err, pred_val_err, overfitting = [], [], []
        tr_accs, va_accs = [], []

        print("Compiling theano train functions...")
        iter_funcs = self._create_iter_functions(y_tensor_type=training_strategy.y_tensor_type,
                                                 objective=training_strategy.objective, learning_rate=learning_rate,
                                                 l_2=training_strategy.L2,
                                                 compute_updates=training_strategy.update_parameters,
                                                 use_weights=training_strategy.use_weights,
                                                 use_mask=training_strategy.use_mask)

        print("Starting training...")
        now = time.time()
        try:

            # initialize early stopping
            last_improvement = 0
            best_model = lasagne.layers.get_all_param_values(self.net)

            # iterate training epochs
            best_va_dice = 0.0
            prev_tr_loss, prev_va_loss = 1e7, 1e7
            prev_acc_tr, prev_acc_va = 0.0, 0.0
            for epoch in self._train(iter_funcs, data_pools, training_strategy.build_train_batch_iterator(),
                                     training_strategy.build_valid_batch_iterator(), training_strategy.report_dices):

                print("Epoch {} of {} took {:.3f}s".format(epoch['number'], training_strategy.max_epochs, time.time() - now))
                now = time.time()

                # --- collect train output ---

                tr_loss, va_loss = epoch['train_loss'], epoch['valid_loss']
                train_acc, valid_acc = epoch['train_acc'], epoch['valid_acc']
                train_dices, valid_dices = epoch['train_dices'], epoch['valid_dices']
                overfit = epoch['overfitting']

                # prepare early stopping
                improvement = va_loss < prev_va_loss

                if improvement:
                    last_improvement = 0
                    best_model = lasagne.layers.get_all_param_values(self.net)
                    best_epoch = epoch['number']

                    # dump net parameters during training
                    if dump_file is not None:
                        with open(dump_file, 'wb') as fp:
                            pickle.dump(best_model, fp)

                last_improvement += 1

                # print train output
                txt_tr = 'costs_tr %.5f ' % tr_loss
                if tr_loss < prev_tr_loss:
                    txt_tr = col.print_colored(txt_tr, BColors.OKGREEN)
                    prev_tr_loss = tr_loss

                txt_tr_acc = '(%.3f)' % train_acc
                if train_acc > prev_acc_tr:
                    txt_tr_acc = col.print_colored(txt_tr_acc, BColors.OKGREEN)
                    prev_acc_tr = train_acc
                txt_tr += txt_tr_acc + ', '

                txt_val = 'costs_val %.5f ' % va_loss
                if va_loss < prev_va_loss:
                    txt_val = col.print_colored(txt_val, BColors.OKGREEN)
                    prev_va_loss = va_loss

                txt_va_acc = '(%.3f)' % valid_acc
                if valid_acc > prev_acc_va:
                    txt_va_acc = col.print_colored(txt_va_acc, BColors.OKGREEN)
                    prev_acc_va = valid_acc
                txt_val += txt_va_acc + ', '

                print('  lr: %.5f' % learn_rate)
                print('  ' + txt_tr + txt_val + 'tr/val %.3f' % overfit)

                # report dice coefficients
                if training_strategy.report_dices:

                    train_str = '  train  |'
                    for key in np.sort(train_dices.keys()):
                        train_str += ' %.2f: %.3f |' % (key, train_dices[key])
                    print(train_str)
                    train_acc = np.max(train_dices.values())

                    valid_str = '  valid  |'
                    for key in np.sort(valid_dices.keys()):
                        txt_va_dice = ' %.2f: %.3f |' % (key, valid_dices[key])
                        if valid_dices[key] > best_va_dice and valid_dices[key] == np.max(valid_dices.values()):
                            best_va_dice = valid_dices[key]
                            txt_va_dice = col.print_colored(txt_va_dice, BColors.OKGREEN)
                        valid_str += txt_va_dice
                    print(valid_str)
                    valid_acc = np.max(valid_dices.values())

                # collect model evolution data
                tr_accs.append(train_acc)
                va_accs.append(valid_acc)
                pred_tr_err.append(tr_loss)
                pred_val_err.append(va_loss)
                overfitting.append(overfit)
                
                # save results
                exp_res = dict()
                exp_res['pred_tr_err'] = pred_tr_err
                exp_res['tr_accs'] = tr_accs
                exp_res['pred_val_err'] = pred_val_err
                exp_res['va_accs'] = va_accs
                exp_res['overfitting'] = overfitting
                
                if log_file is not None:
                    with open(log_file, 'wb') as fp:
                        pickle.dump(exp_res, fp)                
                
                # --- early stopping: preserve best model ---
                if last_improvement > training_strategy.patience:
                    print(col.print_colored("Early Stopping!", BColors.WARNING))
                    status = "Epoch: %d, Best Validation Loss: %.5f: Acc: %.5f" % (
                    best_epoch, prev_va_loss, prev_acc_va)
                    print(col.print_colored(status, BColors.WARNING))

                    if training_strategy.refinement_strategy.n_refinement_steps <= 0:
                        break

                    else:

                        status = "Resetting to best model so far and refining with adopted learn rate."
                        print(col.print_colored(status, BColors.WARNING))

                        # reset net to best weights
                        lasagne.layers.set_all_param_values(self.net, best_model)

                        # update learn rate
                        learn_rate = training_strategy.refinement_strategy.adapt_learn_rate(learn_rate)
                        last_improvement = 0
                        training_strategy.patience = training_strategy.refinement_strategy.refinement_patience

                # maximum number of epochs reached
                if epoch['number'] >= training_strategy.max_epochs:
                    break

                # update learning rate
                learn_rate = training_strategy.adapt_learn_rate(learn_rate, epoch['number'])
                learning_rate.set_value(learn_rate)

        except KeyboardInterrupt:
            pass

        # set net to best weights
        lasagne.layers.set_all_param_values(self.net, best_model)

    def predict_proba(self, input):
        """
        Predict on test samples
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()

        return self.compute_output(*input)

    def predict(self, input):
        """
        Predict class labels on test samples
        """
        return np.argmax(self.predict_proba(input), axis=1)

    def compute_layer_output(self, input, layer):
        """
        Compute output of given layer
        layer: either a string (name of layer) or a layer object
        """

        # prepare input for prediction
        if not isinstance(input, list):
            input = [input]

        # reshape to network input
        if input[0].ndim < len(self.input_shape):
            input[0] = input[0].reshape([1] + list(input[0].shape))

        # get layer by name
        if not isinstance(layer, lasagne.layers.Layer):
            for l in lasagne.layers.helper.get_all_layers(self.net):
                if l.name == layer:
                    layer = l
                    break

        # compile prediction function for target layer
        if layer not in self.compute_output_dict:
            self.compute_output_dict[layer] = self._compile_prediction_function(target_layer=layer)

        return self.compute_output_dict[layer](*input)

    def save(self, file_path):
        """
        Save model to disk
        """
        with open(file_path, 'w') as fp:
            params = lasagne.layers.get_all_param_values(self.net)
            pickle.dump(params, fp, -1)

    def load(self, file_path):
        """
        load model from disk
        """
        with open(file_path, 'rb') as fp:
            params = pickle.load(fp)
        lasagne.layers.set_all_param_values(self.net, params)

    def _compile_prediction_function(self, target_layer=None):
        """
        Compile theano prediction function
        """

        # collect input vars
        all_layers = lasagne.layers.helper.get_all_layers(self.net)
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        # get network output nad compile function
        if target_layer is None:
            target_layer = self.net

        net_output = lasagne.layers.get_output(target_layer, deterministic=True)
        return theano.function(inputs=input_vars, outputs=net_output)

    def _create_iter_functions(self, y_tensor_type, objective, learning_rate, l_2, compute_updates, use_weights, use_mask):
        """ Create functions for training, validation and testing to iterate one epoch. """

        # init target tensor
        targets = y_tensor_type('y')
        weights = y_tensor_type('w').astype("float32")

        # get input layer
        all_layers = lasagne.layers.helper.get_all_layers(self.net)

        # collect input vars
        input_vars = []
        for l in all_layers:
            if isinstance(l, lasagne.layers.InputLayer):
                input_vars.append(l.input_var)

        # compute train costs
        tr_output = lasagne.layers.get_output(self.net, deterministic=False)

        if use_weights or use_mask:
            tr_cost = objective(tr_output, targets, weights)
            tr_input = input_vars + [targets, weights]
        else:
            tr_cost = objective(tr_output, targets)
            tr_input = input_vars + [targets]

        # regularize RNNs
        for l in all_layers:

            # if l.name == "norm_reg_rnn":
            #
            #     H = lasagne.layers.get_output(l, deterministic=False)
            #     H_l2 = T.sqrt(T.sum(H ** 2, axis=-1))
            #     norm_diffs = (H_l2[:, 1:] - H_l2[:, :-1]) ** 2
            #     norm_preserving_loss = T.mean(norm_diffs)
            #
            #     beta = 1.0
            #     tr_cost += beta * norm_preserving_loss

            if l.name == "norm_reg_rnn":

                H = lasagne.layers.get_output(l, deterministic=False)
                steps = T.arange(1, l.output_shape[1])

                def compute_norm_diff(k, H):
                    n0 = ((H[:, k - 1, :]) ** 2).sum(1).sqrt()
                    n1 = ((H[:, k, :]) ** 2).sum(1).sqrt()
                    return (n1 - n0) ** 2

                norm_diffs, _ = theano.scan(fn=compute_norm_diff, outputs_info=None,
                                            non_sequences=[H], sequences=[steps])

                beta = 1.0
                norm_preserving_loss = T.mean(norm_diffs)
                tr_cost += beta * norm_preserving_loss

            else:
                pass

        # compute validation costs
        va_output = lasagne.layers.get_output(self.net, deterministic=True)
        #va_output_stochastic = lasagne.layers.get_output(self.net, deterministic=False)

		# estimate accuracy
        if y_tensor_type == T.ivector:
            va_acc = 100.0 * T.mean(T.eq(T.argmax(va_output, axis=1), targets), dtype=theano.config.floatX)
            tr_acc = 100.0 * T.mean(T.eq(T.argmax(tr_output, axis=1), targets), dtype=theano.config.floatX)
        else:
            va_acc, tr_acc = None, None

        # collect all parameters of net and compute updates
        all_params = lasagne.layers.get_all_params(self.net, trainable=True)

        # add weight decay
        if l_2 is not None:
            all_layers = lasagne.layers.get_all_layers(self.net)
            tr_cost += l_2 * lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2)

        # compute updates
        all_grads = lasagne.updates.get_or_compute_grads(tr_cost, all_params)
        updates = compute_updates(all_grads, all_params, learning_rate)

        # compile iter functions
        tr_outputs = [tr_cost, tr_output]
        if tr_acc is not None:
            tr_outputs.append(tr_acc)
        iter_train = theano.function(tr_input, tr_outputs, updates=updates)

        if use_mask:
            va_inputs = input_vars + [targets, weights]
            va_cost = objective(va_output, targets, weights)
        else:
            va_inputs = input_vars + [targets]
            va_cost = objective(va_output, targets)
        va_outputs = [va_cost, va_output]
        if va_acc is not None:
            va_outputs.append(va_acc)
        iter_valid = theano.function(va_inputs, va_outputs )

        return dict(train=iter_train, valid=iter_valid, test=iter_valid)

    def _train(self, iter_funcs, data_pools, train_batch_iter, valid_batch_iter, estimate_dices):
        """
        Train the model with `dataset` with mini-batch training.
        Each mini-batch has `batch_size` recordings.
        """
        col = BColors()
        from lasagne_wrapper.segmentation_utils import dice

        for epoch in itertools.count(1):

            # evaluate various thresholds
            if estimate_dices:
                threshs = [0.3, 0.4, 0.5, 0.6, 0.7]

                tr_dices = dict()
                for thr in threshs:
                    tr_dices[thr] = []

                va_dices = dict()
                for thr in threshs:
                    va_dices[thr] = []

            else:
                tr_dices = None
                va_dices = None

            # iterate train batches
            batch_train_losses, batch_train_accs = [], []
            iterator = train_batch_iter(data_pools['train'])
            generator = threaded_generator_from_iterator(iterator)

            batch_times = np.zeros(5, dtype=np.float32)
            start, after = time.time(), time.time()
            for i_batch, train_input in enumerate(generator):
                batch_res = iter_funcs['train'](*train_input)
                batch_train_losses.append(batch_res[0])

                # collect classification accuracies
                if len(batch_res) > 2:
                    batch_train_accs.append(batch_res[2])

                # estimate dices for various thresholds
                if estimate_dices:
                    y_b = train_input[1]
                    pred = batch_res[1]
                    for thr in threshs:
                        for i in xrange(pred.shape[0]):
                            seg = pred[i, 0] > thr
                            tr_dices[thr].append(100 * dice(seg, y_b[i, 0]))

                # train time
                batch_time = time.time() - after
                after = time.time()
                train_time = (after - start)

                # estimate updates per second (running avg)
                batch_times[0:4] = batch_times[1:5]
                batch_times[4] = batch_time
                ups = 1.0 / batch_times.mean()

                # report loss during training
                perc = 100 * (float(i_batch) / train_batch_iter.n_batches)
                dec = int(perc // 4)
                progbar = "|" + dec * "#" + (25 - dec) * "-" + "|"
                vals = (perc, progbar, train_time, ups, np.mean(batch_train_losses))
                loss_str = " (%d%%) %s time: %.2fs, ups: %.2f, loss: %.5f" % vals
                print(col.print_colored(loss_str, col.WARNING), end="\r")
                sys.stdout.flush()

            # print("\x1b[K", end="\r")
            print(' ')
            print(' ')
            avg_train_loss = np.mean(batch_train_losses)
            avg_train_acc = np.mean(batch_train_accs) if len(batch_train_accs) > 0 else 0.0
            if estimate_dices:
                for thr in threshs:
                    tr_dices[thr] = np.mean(tr_dices[thr])

            # evaluate classification power of data set

            # iterate validation batches
            batch_valid_losses, batch_valid_accs = [], []
            iterator = valid_batch_iter(data_pools['valid'])
            generator = threaded_generator_from_iterator(iterator)

            for va_input in generator:
                batch_res = iter_funcs['valid'](*va_input)
                batch_valid_losses.append(batch_res[0])

                # collect classification accuracies
                if len(batch_res) > 2:
                    batch_valid_accs.append(batch_res[2])

                # estimate dices for various thresholds
                if estimate_dices:
                    y_b = va_input[1]
                    pred = batch_res[1]
                    for thr in threshs:
                        for i in xrange(pred.shape[0]):
                            seg = pred[i, 0] > thr
                            va_dices[thr].append(100 * dice(seg, y_b[i, 0]))

            avg_valid_loss = np.mean(batch_valid_losses)
            avg_valid_accs = np.mean(batch_valid_accs) if len(batch_valid_accs) > 0 else 0.0
            if estimate_dices:
                for thr in threshs:
                    va_dices[thr] = np.mean(va_dices[thr])

            # collect results
            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'valid_loss': avg_valid_loss,
                'valid_acc': avg_valid_accs,
                'valid_dices': va_dices,
                'train_dices': tr_dices,
                'overfitting': avg_train_loss / avg_valid_loss,
            }


class SegmentationNetwork(Network):
    """
    Segmentation Neural Network
    """
    
    def predict_proba(self, input, squeeze=True):
        """
        Predict on test samples
        """
        if self.compute_output is None:
            self.compute_output = self._compile_prediction_function()
        
        # get network input shape
        l_in = lasagne.layers.helper.get_all_layers(self.net)[0]
        in_shape = l_in.output_shape[-2::]
        
        # standard prediction
        if input.shape[-2::] == in_shape:
            proba = self.compute_output(input)
        
        # sliding window prediction if images do not match
        else:
            proba = self._predict_proba_sliding_window(input)
        
        if squeeze:
            proba = proba.squeeze()
        
        return proba

    def predict(self, input, thresh=0.5):
        """
        Predict label map on test samples
        """
        P = self.predict_proba(input, squeeze=False)
        
        # binary segmentation
        if P.shape[1] == 1:
            return (P > thresh).squeeze()
        
        # categorical segmentation
        else:
            return np.argmax(P, axis=1).squeeze()
        
    
    def _predict_proba_sliding_window(self, images):
        """
        Sliding window prediction for images larger than the input layer
        """
        n_images = images.shape[0]
        h, w = images.shape[2:4]
        _, Nc, sh, sw = self.net.output_shape
        step_h = sh // 2
        step_w = sw // 2
        
        row_0 = np.arange(0, h - step_h, step_h)
        row_1 = row_0 + sh
        shift = h - row_1[-1]
        row_0[-1] += shift
        row_1[-1] += shift
        
        col_0 = np.arange(0, w - step_w, step_w)
        col_1 = col_0 + sw
        shift = w - col_1[-1]
        col_0[-1] += shift
        col_1[-1] += shift     
        
        # initialize result image
        R = np.zeros((n_images, Nc, h, w))
        V = np.zeros((n_images, Nc, h, w))
        
        for ir in xrange(len(row_0)):
            for ic in xrange(len(col_0)):
                I = images[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]]
                # predict on test image
                P = self.predict_proba(I, squeeze=False)
                R[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += P
                V[:, :, row_0[ir]:row_1[ir], col_0[ic]:col_1[ic]] += 1
                
        # normalize predictions
        R /= V
        return R
