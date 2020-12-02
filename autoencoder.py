from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import sketch_rnn

SOS = torch.Tensor([0,0,0,1,0])
EOS = torch.Tensor([0,0,0,0,1])

def load(filename):
    dataset = np.load(filename, encoding='latin1', allow_pickle=True)
    data = dataset['train']
    data = sketch_rnn.purify(data)
    Nmax = sketch_rnn.max_size(data)
    data = [torch.Tensor(x) for x in sketch_rnn.normalize(data)]
    return (data, Nmax)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        if hidden is None:
            hidden = self.initHidden()
        _output, hidden = self.gru(input, hidden)
        return hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden=None):
        #output = F.relu(output) # Everyone loves the relu! why???
        output, hidden = self.gru(input, hidden)
        return self.out(output), hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, input_size)

    def forward(self, input):
        hidden = self.encoder(input, None)
        num_lines = input.shape[1]
        input = SOS.unsqueeze(0).unsqueeze(0)
        outputs = []
        for _ in range(num_lines):
            output, hidden = self.decoder(input, hidden)
            outputs.append(output)
            input = output
        return torch.cat(outputs).transpose(0,1)


        # question: how do we calculate the loss from the decoder, do we compare the whole sequence, and if so, how?
        # yeah we just add them all together

def prepare_input(inp):
    inp = inp.unsqueeze(0)
    inp = F.pad(inp, (0,2))
    inp = torch.cat((SOS.unsqueeze(0).unsqueeze(0), inp), dim=1)
    inp = torch.cat((inp, EOS.unsqueeze(0).unsqueeze(0)), dim=1)
    return inp

class Trainer():
    def __init__(self, input_size, hidden_size=6, learning_rate=0.01):
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, input_size)
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer

    def trainIters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

    def train(input_tensor, target_tensor):
        encoder_hidden = encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length


"""
questions:

in this code:
    def forward(self, input, length):
        input = self.encoder(input)
        decoded_output = self.decoder(input) #Softmax doesnt change a thing here, same problem
        return decoded_output
what are the dimensions of input / decoded_output? is decoded_output the same dimension?
"""
