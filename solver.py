import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Mnist_Classifier(nn.Module):
    def __init__(self):
        super(Mnist_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.fc1 = nn.Linear(7*7*16, 10)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        x = x.view(-1, 7*7*16)

        x = self.fc1(x)

        return x

class Simple_Network(nn.Module):
    def __init__(self):
        super(Simple_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2_2_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x) + F.avg_pool2d(input=self.conv2_2_1(x), kernel_size=(2, 2))
        x = torch.cat(tensors=(self.conv3_2(x), x), dim=1)
        x = self.conv4(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


def dataset(root, train, download):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)


class Solver(object):
    def __init__(self, args):

        self.args = args

        '''Define models'''
        # self.Classifier = Mnist_Classifier()
        self.Classifier = Simple_Network()
        if self.args.cuda:
            self.Classifier.cuda()
        else:
            self.Classifier.cpu()
        '''Define Losses'''
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')

        '''
        Define Optimizers
        '''
        self.opt_Classifier = optim.SGD(self.Classifier.parameters(),
                                        lr=self.args.lr, momentum=self.args.momentum)


        '''
        Load Datasets
        '''
        '''train dataset and dataloader'''
        self.train_dataset = dataset(root=self.args.dataset_dir, train=True, download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                                        drop_last=True, num_workers=0)
        '''test dataset and dataloader'''
        self.test_dataset = dataset(root=self.args.dataset_dir, train=False, download=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, shuffle=False,
                                                       drop_last=True, num_workers=0)


    '''Train'''
    def train(self, epoch):
        device = 'cuda' if self.args.cuda else 'cpu'

        '''Define logs'''
        logs = {}
        logs['classification_loss'] = []
        logs['training_accuracy'] = []

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            '''forward'''
            logits = self.Classifier(inputs)

            '''accuracy check'''
            pred = logits.argmax(dim=1, keepdim=True)
            accuracy = pred.eq(targets.view_as(pred)).sum().item() / self.args.test_batch_size
            logs['training_accuracy'].append(accuracy)

            '''compute loss'''
            classification_loss = self.CrossEntropyLoss(logits, targets)
            logs['classification_loss'].append(classification_loss.item())

            '''compute gradients'''
            classification_loss.backward()

            '''optimize parameters'''
            self.opt_Classifier.step()

            '''zero gradient'''
            self.opt_Classifier.zero_grad()

        return logs

    def test(self):
        self.Classifier.eval()
        device = 'cuda' if self.args.cuda else 'cpu'

        '''Define logs'''
        logs = {}
        logs['test_loss'] = []
        logs['test_accuracy'] = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = self.Classifier(inputs)

                classificaiton_loss =  self.CrossEntropyLoss(logits, targets)

                pred = logits.argmax(dim=1, keepdim=True)
                accuracy = pred.eq(targets.view_as(pred)).sum().item() / self.args.test_batch_size

                logs['test_loss'].append(classificaiton_loss)
                logs['test_accuracy'].append(accuracy)

        return logs


    def save_model(self, path):
        torch.save(self.Classifier.state_dict(), self.args.checkpoint_dir + '/' + path + '/Classifier.pth')

    def load_model(self, path):
        self.Classifier.load_state_dict(
            torch.load(self.args.checkpoint_dir + '/' + path +'/Classifier.pth'))

