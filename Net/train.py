import file_reader as fr
import os.path

from Net.net import Net
from Net.readData import get_data_set

from torch import nn, optim

IMAGE_SIZE = 64  # 图像尺寸
LABEL_BYTES = 8 * 8  # 标签大小

SAMPLE_LENGTH = 64 * 64 + 8 * 8  # 每个样本尺寸
data_dir = 'Data/'  # path of training/validation/test data
trainPath = 'AI_Train_27300.dat'

BATCH_SIZE = 100
ITER_TIMES = 1000000

fileReader = fr.FileReader()
fileReader.initialize(os.path.join(data_dir, trainPath), BATCH_SIZE * SAMPLE_LENGTH)
data = get_data_set(fileReader, BATCH_SIZE * SAMPLE_LENGTH)

net = Net()
print(i for i in net.parameters())
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

lossFunc = nn.CrossEntropyLoss()

for epoch in range(ITER_TIMES):  # loop over the dataset multiple times
    for i in range(50):
        batch = data.next_batch(BATCH_SIZE)

        img = batch[0]
        label = batch[1]

        out = net(img)
        # print(out.shape)
        loss = lossFunc(out, label)
        loss.backward()
        optimizer.step()
        print('loss->',loss)
