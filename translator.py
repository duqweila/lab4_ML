import re

import torch

from data_preprocessing import prepareData, MAX_LENGTH
from model import EncoderRNN, AttnDecoderRNN, device, SOS_token, EOS_token
from train import tensorFromSentence

hidden_size = 256
input_lang, output_lang, pairs = prepareData('eng', 'cmn', True)
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
encoder.load_state_dict(torch.load('encoder.pth'))
attn_decoder.load_state_dict(torch.load('decoder.pth'))
encoder = encoder.eval()
attn_decoder = attn_decoder.eval()


def external_translate(input_sentence):
    # Translate the input sentence
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, input_sentence)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_tensor.size()[0]):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = attn_decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return ' '.join(decoded_words)