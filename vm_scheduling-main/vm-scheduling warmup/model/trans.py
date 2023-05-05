import logging
from .components.transformer import *
from .components.loss import *
from torch.nn.modules.container import ModuleList

logger = logging.getLogger('VM.enc_dec_attn')


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        self.device = params.device
        self.batch_size = params.batch_size
        self.seq_length = params.num_pm

        self.pm_encode = nn.Linear(params.pm_cov, params.transformer_input_size)
        self.vm_encode = nn.Linear(params.vm_cov, params.transformer_input_size)

        encoder_layer = TransformerEncoderLayer(params)
        self.encoder = TransformerEncoder(encoder_layer, params)
        decoder_layer = TransformerDecoderLayer(params)
        self.decoder = TransformerDecoder(decoder_layer, params)

        if params.output_categorical:
            self.quantiles = params.quantiles
            self.output_layer = nn.Linear(params.transformer_input_size, len(self.quantiles) - 1)
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.output_layer = nn.Linear(params.transformer_input_size, 1)
            self.loss_fn = nn.L1Loss()
        self.output_categorical = params.output_categorical

    def forward(self, vm_states, pm_states, return_attns=False):
        vm_encode = self.vm_encode(vm_states)
        pm_encode = self.pm_encode(pm_states)
        memory = self.encoder(output=pm_encode)
        if return_attns:
            output, attns = self.decoder(output=vm_encode, memory=memory, return_attns=True)
            print('output: ', output.shape)
            print('attn: ', attns.shape)
            score = self.output_layer(output[:, 0])
            return score, attns[:, 0]
        else:
            output = self.decoder(output=vm_encode, memory=memory)
            print('output: ', output.shape)
            score = self.output_layer(output[:, 0])
            return score

    def do_train(self, vm_states, pm_states, labels):
        """ Train for one batch.
        Args:
            vm_states ([batch_size, num_vm, vm_cov]): features of virtual machines
            pm_states ([batch_size, num_pm, pm_cov]): features of physical machines
            labels ([batch_size, 1]): ground truth of target
        Returns:
            loss_list (list): list of each individual loss component for plotting
            loss (int): total loss for backpropagation
        """
        predict = self(vm_states, pm_states)
        loss = self.loss_fn(predict, labels)
        loss.backward()
        return [loss.item()], loss

    def test(self, vm_states, pm_states, labels):
        """ Test for one batch.
        Args:
            vm_states ([batch_size, num_vm, vm_cov]): features of virtual machines
            pm_states ([batch_size, num_pm, pm_cov]): features of physical machines
            labels ([batch_size, 1]): ground truth of target
        Returns:
            sampled_params ([batch_size, predict_steps, num_quantiles]): return quantiles as specified in params.json.
            enc_self_attention ([batch_size, predict_steps, predict_start]): to visualize encoder-decoder attention.
        """
        if self.output_categorical:
            logits, attn = self(vm_states, pm_states, return_attns=True)
            predict = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            predict, attn = self(vm_states, pm_states, return_attns=True)
        return {
            'predictions': predict,
            'enc_dec_attn': attn
        }
