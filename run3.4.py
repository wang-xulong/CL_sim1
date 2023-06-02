"""
calculate the functional similarity(different situation) and acc, fgt for CIFAR100 2 class
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import resnet50
import wandb
import datetime
from argparse import Namespace

from util import trainES, get_Cifar100

# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
config = Namespace(
    project_name='CIFAR10',
    basic_task=1,  # count from 0
    experience=5,
    train_bs=128,
    test_bs=128,
    lr_init=0.001,
    max_epoch=2000,
    run_times=10,
    patience=50,
    class_num=2
)
notes = "run model for compare similarity measure"


# use GPU?
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

wandb.init(project=config.project_name, config=config.__dict__, name=now_time,
           save_code=True, notes=notes)

# ------------------------------------ step 1/5 : load data------------------------------------
train_stream, test_stream = get_Cifar100(train_bs=config.train_bs, test_bs=config.test_bs)
# ------------------------------------ step 2/5 : define network-------------------------------
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, config.class_num)
# ------------------------------------ step 3/5 : define loss function and optimization ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr_init)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
# ------------------------------------ step 4/5 : training --------------------------------------------------
# training basic task
basic_task_data = train_stream[config.basic_task]
basic_task_test_data = test_stream[config.basic_task]
model, avg_train_losses, _, _, _, _ = trainES(basic_task_data, basic_task_test_data, model,
                                              criterion, optimizer,
                                              config.max_epoch, device, patience=config.patience,
                                              task_id=0, func_sim=False)


# pop the src data from train_stream and test_stream
train_stream.pop(config.basic_task)
test_stream.pop(config.basic_task)

# save task 1
PATH = "./"
trained_model_path = os.path.join(PATH, 'basic_model.pth')
torch.save(model.state_dict(), trained_model_path)


for j, (train_data, test_data) in enumerate(zip(train_stream, test_stream)):
    print("task {} starting...".format(j+1))
    # load old task's model
    trained_model = resnet50()
    trained_model.fc = nn.Linear(trained_model.fc.in_features, config.class_num)  # final output dim = 2
    trained_model.load_state_dict(torch.load(trained_model_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(trained_model.parameters(), lr=config.lr_init, momentum=0.9, dampening=0.1)
    # training other tasks
    trained_model, _, _, _, _, new_task_loss = trainES(train_data, test_data, trained_model, criterion,
                                                       optimizer, config.max_epoch,
                                                       device, config.patience,
                                                       task_id=j + 1, func_sim=False)
    trained_model_path = os.path.join(PATH, 'model_{}.pth'.format(j+1))
    torch.save(trained_model.state_dict(), trained_model_path)


