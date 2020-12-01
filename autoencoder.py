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
        print(input.shape)
        if hidden is None:
            hidden = self.initHidden()
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def prepare_input(inp):
    inp = inp.unsqueeze(0)
    inp = F.pad(inp, (0,2))
    sos = torch.Tensor([0,0,0,1,0])
    eos = torch.Tensor([0,0,0,0,1])
    inp = torch.cat((sos.unsqueeze(0).unsqueeze(0), inp), dim=1)
    inp = torch.cat((inp, eos.unsqueeze(0).unsqueeze(0)), dim=1)
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


