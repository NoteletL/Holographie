"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
"""

import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod
import datetime
import matplotlib.pyplot as plt
import numpy as np
from .utils import save_images
# import utils

toto = 4


class NeuralNetwork(nn.Module, ABC):
    """An abstract class representing a neural network.

    All other neural network should subclass it. All subclasses should override
    ``forward``, that makes a prediction for its input argument, and
    ``criterion``, that evaluates the fit between a prediction and a desired
    output. This class inherits from ``nn.Module`` and overloads the method
    ``named_parameters`` such that only parameters that require gradient
    computation are returned. Unlike ``nn.Module``, it also provides a property
    ``device`` that returns the current device in which the network is stored
    (assuming all network parameters are stored on the same device).
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        # This is important that this is a property and not an attribute as the
        # device may change anytime if the user do ``net.to(newdevice)``.
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class NNRegressor(NeuralNetwork):
    """
    This class represent an abstract neural network
    """

    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = torch.nn.MSELoss()

    def criterion(self, y ,d):
        """
        This method return a float that evaluation the accuracy of the network

        Arguments:
            y (torch.Tensor)    : The predicted noise free reference
            d (torch.Tensor)    : The clean reference
        """
        return self.mse(y, d)


class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        if(self.number_update == 0):
            return self.running_loss
        return self.running_loss / self.number_update


class DenoisingStatsManager(StatsManager):
    """
    This class manage the stats of an experiment
    """

    def __init__(self):
        super(DenoisingStatsManager, self).__init__()


    def init(self):
        super(DenoisingStatsManager, self).init()
        self.running_psnr = 0


    def accumulate(self, loss, x, y, d):
        """
        This method add new results for the stats manager

        Arguments:
            loss (???)
            x (torch.Tensor)    : The noisy reference
            y (torch.Tensor)    : The predicted noise free reference
            d (torch.Tensor)    : The clean reference
        """
        #print("test accumulate")
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)
        n = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        self.running_psnr += 10*torch.log10(4*n/(torch.norm(y-d)**2))


    def summarize(self):
        """
        This method return the actual stats managed by the stats manager
        """
        loss = super(DenoisingStatsManager, self).summarize()
        psnr = self.running_psnr / self.number_update if(self.number_update !=0) else self.running_psnr
        return {'loss': loss, 'PSNR': psnr}



class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    def __init__(self, net, optimizer, stats_manager, startEpoch=None,
                 input_dir=None, output_dir=None, perform_validation_during_training=False, freq_save=1):

        # Initialize history
        history = []

        # Define checkpoint paths
        if input_dir is None:
            input_dir = './PyTorchCheckpoint/'

        if output_dir is None:
            output_dir = input_dir + '/experiment_{}'.format(datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S"))
        else:
            output_dir = input_dir + '{}'.format(output_dir)


        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(input_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)


        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
        #    with open(config_path, 'r') as f:
        #        if f.read()[:-1] != repr(self):
        #            raise ValueError(
        #                "Cannot create this experiment: "
        #                "I found a checkpoint conflicting with the current setting.")
            self.load(startEpoch)

    def initData(self, train_set, val_set, batch_size=16):
        """This method is used to initialize the training and evaluation data

        Arguments:
            train_set ()                : The training dataset
            val_set ()                  : The evaluation dataset
            batch_size (int, optionnal) : The size of the batch for training data
        """

        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        # Define data loaders
        self.train_loader = td.DataLoader(train_set, batch_size=batch_size, num_workers=0,shuffle=True, drop_last=True, pin_memory=True)
        self.val_loader = td.DataLoader(val_set, batch_size=1, num_workers=0, shuffle=False, drop_last=True, pin_memory=True)

        self.training_data = train_set.getTrainingName()


    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Net': self.net,
                'TrainSet': self.train_set,
                'ValSet': self.val_set,
                'Optimizer': self.optimizer,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training,
                'Training_data': self.training_data}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': self.net.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint_{:0>5}.pth.tar".format(self.epoch))
        self.config_path = os.path.join(self.output_dir, "config.txt")

        torch.save(self.state_dict(), self.checkpoint_path)
        torch.save(self.state_dict(), os.path.join(self.output_dir, "last_checkpoint.pth.tar"))

        with open(self.config_path, 'w') as f:
            print(self, file=f)


    def save_train(self, timeElapsed):
        """This method is used to show and save the unfolding of the training
        """

        self.train_path = os.path.join(self.output_dir, "train.txt")

        print("Epoch {} | Time: {:.2f}s | Training Loss: {:.6f} | Evaluation Loss: {:.6f}".format(
            self.epoch, timeElapsed, self.history[-1][0]['loss'], self.history[-1][1]['loss']))


        with open(self.train_path, 'a') as f:
            print("Epoch {} | Time: {:.2f}s | Training Loss: {:.6f} | Evaluation Loss: {:.6f}".format(
                self.epoch, timeElapsed, self.history[-1][0]['loss'], self.history[-1][1]['loss']), file=f)






    def load(self, epoch=None):
        """Loads the experiment from the given epoch's checkpoint saved on disk.

        Arguments:
            epoch (integer, optional): the number from wich to resume training
        """

       # with open(os.path.join(self.input_dir, "state.txt"), 'r') as f:
       #     print(f.read()[:-1])

        print("epoch", epoch)
        if epoch is None:
            checkpoint = torch.load(os.path.join(self.input_dir, "last_checkpoint.pth.tar"), map_location=self.net.device)
        else:
            checkpoint = torch.load(os.path.join(self.input_dir, "checkpoint_{:0>5}.pth.tar".format(epoch)), map_location=self.net.device)

        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.save()
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
        s = time.time()
        for epoch in range(start_epoch, num_epochs):
            self.stats_manager.init()
            for x, d in self.train_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)
                self.optimizer.zero_grad()
                y = self.net.forward(x)
                loss = self.net.criterion(y, d)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    self.stats_manager.accumulate(loss.item(), x, y, d)
            if not self.perform_validation_during_training:
                self.history.append((self.stats_manager.summarize(), {'loss': 0}))
            else:
                self.history.append((self.stats_manager.summarize(), self.evaluate()))

            self.save_train(time.time() - s)

            if((self.epoch % self.freq_save) == 0):
                self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.net.eval()
        with torch.no_grad():
            for x, d in self.val_loader: # (xSin, xCos)
                xSin = torch.sin(x)
                xCos = torch.cos(x)
                xSin, xCos, d = xSin.to(self.net.device), xCos.to(self.net.device), d.to(self.net.device)
                ySin = self.net.forward(xSin)
                yCos = self.net.forward(xCos)
                x = torch.angle(xCos + xSin * 1J)
                y = torch.angle(yCos + ySin * 1J)
                save_images(os.path.join('.', 'test.tiff'), y.cpu().numpy().squeeze())
                loss = self.net.criterion(y, d)
                self.stats_manager.accumulate(loss.item(), x, y, d)
        self.net.train()
        return self.stats_manager.summarize()

    def getConfig():
        # config = "null"

        with open(os.path.join(self.input_dir, 'config.txt'), 'r') as f:
            config = f.read()[:-1]

        return config


    def test(self, noisy):

        noisy = noisy.to(self.net.device)

        self.net.eval()
        clean_pred_rad = self.net.forward(noisy)
        self.net.train()

        return clean_pred_rad



    def trace(self):
        loss_tab = []
        for k,v in (self.history):
            loss_tab = np.append(loss_tab,round(k['loss'],6))

        print("affichage graphique loss: ")
        plt.plot(np.arange(0,len(loss_tab)),loss_tab)
        plt.title("Losses/epoch Graph ")
        plt.xlabel("Nb Epoch")
        plt.ylabel("Nb Losses")
        plt.show()
