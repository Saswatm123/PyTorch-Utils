import torch
from torch import nn, optim

class MonteCarloDropout(nn.Dropout):
    '''
        Class to use in place of Dropout if Monte Carlo Dropout is wanted instead.
        For example:
            nn.Sequential(
                nn.Linear(...),
                nn.ReLU(...),
                nn.MonteCarloDropout(p=.5)
            )
        To perform inference with it, train model as usual first, then pass model as argument to
        MonteCarloNetHandler.
    '''
    def forward(self, X):
        prev_training_state = self.training
        self.train()
        result = super().forward(X) * (1-self.p)
        self.train(mode=prev_training_state)
        return result

    def train(self, mode=True):
        self.training = mode

class MonteCarloNetHandler:
    '''
        Gets passed in a trained neural network that contains MonteCarloDropout layers in it, 
        performs the inference logic (running it N times and returning various output types, etc.)
    '''
    def __init__(self, net, n):
        self.net = net
        self.n = n

    def __call__(self, X, raw_output = False):
        '''
            Desc:
                Computes Monte Carlo Dropout model by running it N times, where N
                is defined in the initializer. Can toggle between returning all predictions
                and just returning the max prediction & confidence. Max prediction & confidence
                is picked naively here, where we average the confidences & pick the element with the
                highest avg confidence.
            Args:
                X:
                    Data batch of shape [batchsize, data_dimensions...]
                raw_output:
                    If true, we return the raw prediction values for every net as a large tensor. If false,
                    we simply return the prediction with highest confidence and its confidence value.
            Returns:
                Depending on raw_output, we can return either the entire prediction tensor or just the highest confidence
                prediction & its confidence value.
        '''
        with torch.no_grad():
            net.eval()
            results = torch.cat( list(self.net(X) for _ in range(self.n) ) )
            if raw_output:
                return results
            results = results.mean(dim=0)
            return results.argmax(), results.max()