"""
calculate the functional similarity(different situation) and acc, fgt for CIFAR100 2 class
"""
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.models.resnet import resnet50
import wandb
import datetime
from argparse import Namespace

from metrics import compute_acc_fgt, single_run_avg_end_fgt, single_run_avg_end_acc
from util import trainES, get_Cifar100, test

# ------------------------------------ step 0/5 : initialise hyper-parameters ------------------------------------
config = Namespace(
    project_name='CIFAR100',
    basic_task=1,  # count from 0
    experience=5,
    train_bs=128,
    test_bs=200,
    lr_init=0.001,
    max_epoch=2000,
    run_times=10,
    patience=50,
    class_num=2
)
notes = "数据集是 Cifar100_2class_v3，测试模型只log记录1次, basic task = 1"
loss_num = 7
accuracy_list1 = []  # multiple run
accuracy_list2 = []
accuracy_list3 = []
accuracy_list4 = []
fun_score_list = []
# use GPU?
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
for run in range(config.run_times):
    wandb.init(project=config.project_name, config=config.__dict__, name=now_time + "run:" + str(run + 1),
               save_code=True, notes=notes)
    print("run time: {}".format(run + 1))

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
    # 记录本次每个任务的fun_score
    fun_score = np.zeros((4, (loss_num + 1)))  # 4个任务，7个task loss 和一个 basic loss
    # 记录basic loss  4个任务
    fun_score[:, 0] = avg_train_losses[-1]
    # print("basic loss:{:.4}".format(basic_loss))

    # setting stage 1 matrix
    acc_array1 = np.zeros((4, 2))
    # testing basic task
    _, acc_array1[:, 0] = test(test_stream[config.basic_task], model, criterion, device, task_id=0)
    # pop the src data from train_stream and test_stream
    train_stream.pop(config.basic_task)
    test_stream.pop(config.basic_task)
    # test other tasks except basic task
    for i, probe_data in enumerate(test_stream):
        with torch.no_grad():
            _, acc_array1[i, 1] = test(probe_data, model, criterion, device, task_id= i+1)
    # save task 1
    PATH = "./"
    trained_model_path = os.path.join(PATH, 'basic_model.pth')
    torch.save(model.state_dict(), trained_model_path)

    # setting stage 2 matrix
    acc_array2 = np.zeros((4, 2))
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
                                                           task_id=j + 1, func_sim=True)
        # record func_sim of current new task
        for index in range(1, 1 + loss_num):
            fun_score[j, index] = new_task_loss[index - 1]

        # test model on basic task and task j
        with torch.no_grad():
            _, acc_array2[j, 0] = test(basic_task_test_data, trained_model, criterion, device, task_id=0)
            _, acc_array2[j, 1] = test(test_stream[j], trained_model, criterion, device, task_id=j+1)
        # computing avg_acc and CF
    accuracy_list1.append([acc_array1[0, :], acc_array2[0, :]])
    accuracy_list2.append([acc_array1[1, :], acc_array2[1, :]])
    accuracy_list3.append([acc_array1[2, :], acc_array2[2, :]])
    accuracy_list4.append([acc_array1[3, :], acc_array2[3, :]])
    fun_score_list.append(fun_score)
    # 打印每次的end_acc和fgt  # 针对4个任务
    print("task 1 end acc is {}".format(single_run_avg_end_acc(accuracy_list1[run])))
    print("task 1 fgt is {}".format(single_run_avg_end_fgt(accuracy_list1[run])))
    print("task 2 end acc is {}".format(single_run_avg_end_acc(accuracy_list2[run])))
    print("task 2 fgt is {}".format(single_run_avg_end_fgt(accuracy_list2[run])))
    print("task 3 end acc is {}".format(single_run_avg_end_acc(accuracy_list3[run])))
    print("task 3 fgt is {}".format(single_run_avg_end_fgt(accuracy_list3[run])))
    print("task 4 end acc is {}".format(single_run_avg_end_acc(accuracy_list4[run])))
    print("task 4 fgt is {}".format(single_run_avg_end_fgt(accuracy_list4[run])))

    wandb.finish()
accuracy_array1 = np.array(accuracy_list1)
accuracy_array2 = np.array(accuracy_list2)
accuracy_array3 = np.array(accuracy_list3)
accuracy_array4 = np.array(accuracy_list4)

avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array1)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array2)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array3)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array4)
print('----------- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))

# save func_sim and metrics
for t in range(config.run_times):
    np.savetxt("func_score of time " + str(t) + ".csv", fun_score_list[t], delimiter=',')
