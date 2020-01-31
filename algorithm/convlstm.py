import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.device = device

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # These weight will be initialized when the first batch example comes in according to its shape
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1]).to(self.device)
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1]).to(self.device)
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1]).to(self.device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        self.h_0 = torch.zeros(batch_size, hidden, shape[0], shape[1]).to(self.device)
        self.c_0 = torch.zeros(batch_size, hidden, shape[0], shape[1]).to(self.device)
        return torch.zeros(batch_size, hidden, shape[0], shape[1]).to(self.device),\
               torch.zeros(batch_size, hidden, shape[0], shape[1]).to(self.device)


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, device, step=1):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, device)
            setattr(self, name, cell)
            self._all_layers.append(cell)

        self.reset_memory()

    def forward(self, input):
        # outputs = []
        internal_state = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                # do forward
                if self.internal_state[i] == (None, None):
                    bsize, _, height, width = x.size()
                    h, c = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    self.internal_state[i] = (h, c)

                h, c = self.internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                self.internal_state[i] = (x, new_c)
            # output.append((x, new_c))
        return x, new_c

    def reset_memory(self):
        self.internal_state = [(None, None) for _ in range(self.num_layers)]



if __name__ == '__main__':
    # gradient check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    convlstm = ConvLSTM(input_channels=7, hidden_channels=[5, 3, 3], kernel_size=3, device=device, step=3)
    loss_fn = torch.nn.MSELoss()

    input = torch.randn(1, 7, 10, 10)
    target = torch.randn(1, 3, 10, 10).double()

    output = convlstm(input)
    output = output[0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)