import torch
torch.manual_seed(10)
from torch.autograd import Variable
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_path import GENRES, DATAPATH, MODELPATH
from model import genreNet
from raw_data import Data
from data_set_split import Set
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import time

time_program = time.time()

def main(rank, world_size):


    data = Data(GENRES, DATAPATH)
    data.make_raw_data()
    data.save()
    data = Data(GENRES, DATAPATH)
    data.load()

    set_ = Set(data)
    set_.make_dataset()
    set_.save()
    set_ = Set(data)
    set_.load()

    x_train, y_train = set_.get_train_set()
    x_valid, y_valid = set_.get_valid_set()
    x_test,  y_test = set_.get_test_set()

    TRAIN_SIZE = len(x_train)
    VALID_SIZE = len(x_valid)
    TEST_SIZE = len(x_test)

    EPOCH_NUM = 100
    BATCH_SIZE = 256

    setup(rank, world_size)

    model = genreNet().to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(ddp_model.parameters(), lr=1e-4)

    for epoch in range(EPOCH_NUM):
        
        inp_train, out_train = Variable(torch.from_numpy(x_train)).float().to(rank), Variable(torch.from_numpy(y_train)).long().to(rank)
        inp_valid, out_valid = Variable(torch.from_numpy(x_valid)).float().to(rank), Variable(torch.from_numpy(y_valid)).long().to(rank)
        train_loss = 0
        optimizer.zero_grad()  # <-- OPTIMIZER
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            x_train_batch, y_train_batch = inp_train[i:i + BATCH_SIZE], out_train[i:i + BATCH_SIZE]

            pred_train_batch = ddp_model(x_train_batch)
            loss_train_batch = criterion(pred_train_batch, y_train_batch)
            train_loss += loss_train_batch.data.cpu().numpy()

            loss_train_batch.backward()
        optimizer.step()  # <-- OPTIMIZER

        epoch_train_loss = (train_loss * BATCH_SIZE) / TRAIN_SIZE
        train_sum = 0
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            pred_train = ddp_model(inp_train[i:i + BATCH_SIZE])
            indices_train = pred_train.max(1)[1]
            train_sum += (indices_train == out_train[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
        train_accuracy = train_sum / float(TRAIN_SIZE)

        valid_loss = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            x_valid_batch, y_valid_batch = inp_valid[i:i + BATCH_SIZE], out_valid[i:i + BATCH_SIZE]

            pred_valid_batch = ddp_model(x_valid_batch)
            loss_valid_batch = criterion(pred_valid_batch, y_valid_batch)
            valid_loss += loss_valid_batch.data.cpu().numpy()

        epoch_valid_loss = (valid_loss * BATCH_SIZE) / VALID_SIZE
        valid_sum = 0
        for i in range(0, VALID_SIZE, BATCH_SIZE):
            pred_valid = ddp_model(inp_valid[i:i + BATCH_SIZE])
            indices_valid = pred_valid.max(1)[1]
            valid_sum += (indices_valid == out_valid[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
        valid_accuracy = valid_sum / float(VALID_SIZE)

        #print("Epoch: %d\t\tTrain loss : %.2f\t\tValid loss : %.2f\t\tTrain acc : %.2f\t\tValid acc : %.2f" % \
        #      (epoch + 1, epoch_train_loss, epoch_valid_loss, train_accuracy, valid_accuracy))
        
    torch.save(ddp_model.state_dict(), MODELPATH)
    print('\nProgram Out -> pyTorch model is saved.')

    inp_test, out_test = Variable(torch.from_numpy(x_test)).float().to(rank), Variable(torch.from_numpy(y_test)).long().to(rank)
    test_sum = 0
    for i in range(0, TEST_SIZE, BATCH_SIZE):
        pred_test = ddp_model(inp_test[i:i + BATCH_SIZE])
        indices_test = pred_test.max(1)[1]
        test_sum += (indices_test == out_test[i:i + BATCH_SIZE]).sum().data.cpu().numpy()
    test_accuracy = test_sum / float(TEST_SIZE)

    print("Test acc: %.2f\n" % (test_accuracy*100))
    print("Total Time:", convert(time.time() - time_program))

    #cleanup()

    return 

def cleanup():
    dist.destroy_process_group()


def run(train, world_size):
    mp.spawn(train, args= (world_size,), nprocs = world_size, join = True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def convert(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


if __name__ == '__main__':
    if torch.cuda.is_available():
        n_gpu = 3#torch.cuda.device_count()
        print('\nRunning on ', n_gpu, 'GPUs')

        run(main, n_gpu)

    else:
        print('\nNo Cuda GPU available')

"""Reference: 1.https://stackoverflow.com/questions/65179954/should-a-data-batch-be-moved-to-cpu-and-converted-from-torch-tensor-to-a-numpy 2. https://deeplizard.com/learn/video/Bs1mdHZiAS8"""





