import argparse
import functools
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchsummary import summary

from data_utils.noise_perturb import NoisePerturbAugmentor
from data_utils.reader import CustomDataset, collate_fn
from data_utils.speed_perturb import SpeedPerturbAugmentor
from data_utils.volume_perturb import VolumePerturbAugmentor
from modules.ecapa_tdnn import EcapaTdnn
from utils.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('audio_duration',   float,  3,                        '训练的音频长度，单位秒')
add_arg('min_duration',     float,  0.5,                      '训练的最短音频长度，单位秒')
add_arg('num_epoch',        int,    10,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_list_path',  str,    'dataset/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'dataset/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model_dir',   str,    'output/models/',         '模型保存的路径')
add_arg('feature_method',   str,    'melspectrogram',         '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
add_arg('augment_conf_path',str,    'configs/augment.yml',    '数据增强的配置文件，为json格式')
add_arg('resume',           str,    None,                     '恢复训练的模型文件夹，当为None则不使用恢复模型')
args = parser.parse_args()


# 评估模型
@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    accuracies, preds, labels = [], [], []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.numpy()
        output = model(spec_mag)
        output = output.data.cpu().numpy()
        pred = np.argmax(output, axis=1)
        preds.extend(pred.tolist())
        labels.extend(label.tolist())
        acc = np.mean((pred == label).astype(int))
        accuracies.append(acc.item())
    model.train()
    acc = float(sum(accuracies) / len(accuracies))
    return acc


def train(args):
    # 获取数据增强器
    augmentors = None
    if args.augment_conf_path is not None:
        augmentors = {}
        with open(args.augment_conf_path, encoding="utf-8") as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
        augmentors['noise'] = NoisePerturbAugmentor(**configs['noise'])
        augmentors['speed'] = SpeedPerturbAugmentor(**configs['speed'])
        augmentors['volume'] = VolumePerturbAugmentor(**configs['volume'])
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path,
                                  feature_method=args.feature_method,
                                  mode='train',
                                  sr=16000,
                                  chunk_duration=args.audio_duration,
                                  min_duration=args.min_duration,
                                  do_vad=False,
                                  augmentors=augmentors)
    """ DataLoader用法
        -　使用流程
            ① 创建一个 Dataset 对象
            ② 创建一个 DataLoader 对象
            ③ 循环这个 DataLoader 对象
        -　参数
            dataset(Dataset): 传入的数据集
            batch_size(int, optional): 每个batch有多少个样本
            shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
            sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
            batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
            num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
            collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
            pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
            drop_last (bool, optional): 如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
            timeout(numeric, optional): 如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0
            worker_init_fn (callable, optional): 每个worker初始化函数 If not None, this will be called on each
    """
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)

    test_dataset = CustomDataset(args.test_list_path,
                                 feature_method=args.feature_method,
                                 mode='eval',
                                 sr=16000,
                                 do_vad=False,
                                 chunk_duration=args.audio_duration)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)
    # 获取模型
    device = torch.device("cuda")
    if args.use_model == 'ecapa_tdnn':
        model = EcapaTdnn(num_classes=args.num_classes, input_size=train_dataset.input_size)
    else:
        raise Exception(f'{args.use_model} 模型不存在!')
    model.to(device)

    # torchsummary 能够查看模型的输入和输出的形状，可以更加清楚地输出模型的结构
    # torchsummary.summary(model, input_size, batch_size=-1, device="cuda")
    """
        model：pytorch 模型，必须继承自 nn.Module
        input_size：模型输入 size，形状为 C，H ，W
        batch_size：batch_size，默认为 -1，在展示模型每层输出的形状时显示的 batch_size
        device："cuda"或者"cpu"，默认device=‘cuda’
    """
    summary(model, (train_dataset.input_size, 98))

    # 获取优化方法
    # 优化器的作用：求出让损失函数最小化的参数
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=5e-4)
    # 获取学习率衰减函数，它的作用是在训练的过程中，对学习率的值进行衰减，训练到达一定程度后，使用小的学习率来提高精度。
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    # 恢复训练
    last_epoch = 0
    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model.pth')))
        state = torch.load(os.path.join(args.resume, 'model.state'))
        last_epoch = state['last_epoch']
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        print(f'成功加载第 {last_epoch} 轮的模型参数和优化方法参数')

    # 获取损失函数
    loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数，刻画的是实际输出（概率）与期望输出（概率）的距离。https://zhuanlan.zhihu.com/p/98785902

    # train_loader的长度是根据数据的总数和batch_size的大小共同计算出来的
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    # 开始训练
    for epoch in range(args.num_epoch):
        loss_sum = []
        accuracies = []
        train_times = []
        start = time.time()

        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        # 语法为　enumerate(sequence, [start=0])
        """参数
            sequence -- 一个序列、迭代器或其他支持迭代对象。
            start -- 下标起始位置的值。
        """
        for batch_id, (spec_mag, label) in enumerate(train_loader):
            spec_mag = spec_mag.to(device)
            label = label.to(device).long()
            output = model(spec_mag)
            # 计算损失值
            los = loss(output, label)
            # 因为grad在反向传播的过程中是累加的，也就是说上一次反向传播的结果会对下一次的反向传播的结果造成影响，则意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零。
            optimizer.zero_grad()  # 梯度置零
            los.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 计算准确率
            # softmax() 将各个输出节点的输出值范围映射到[0, 1]，并且约束各个输出节点的输出值的和为1
            # dim 0 - 对某一维度中的列进行softmax；1 - 对某一维度的行进行softmax；-1 - 对某一维度的行进行softmax
            """
                import numpy as np
    
                a = np.asarray([1, 2, 3, 4])
                b = np.asarray([1, 5, 6, 7])
                c = (a == b).astype(int)
                mean = np.mean(c)
                print(a)
                print(b)
                print(c)
                print(mean)
                
                [1 2 3 4]
                [1 5 6 7]
                [1 0 0 0]
                0.25
            """
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            train_times.append((time.time() - start) * 1000)
            if batch_id % 100 == 0:
                eta_sec = (sum(train_times) / len(train_times)) * (sum_batch - (epoch - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print(f'[{datetime.now()}] Train epoch [{epoch}/{args.num_epoch}], batch: {batch_id}/{len(train_loader)}, '
                      f'lr: {scheduler.get_last_lr()[0]:.8f}, loss: {sum(loss_sum) / len(loss_sum):.8f}, '
                      f'accuracy: {sum(accuracies) / len(accuracies):.8f}, '
                      f'eta: {eta_str}')  # eta: Estimated Time of Arrival - 预计到达时间
            start = time.time()
        scheduler.step()
        # 评估模型
        acc = evaluate(model, test_loader, device)
        print('='*70)
        print(f'[{datetime.now()}] Test {epoch}, Accuracy: {acc}')
        print('='*70)
        # 保存模型
        """torch.save
            - torch.save(model,'net.pth') 保存加载整个模型（不推荐）
            - torch.save(model.state_dict(),'net_params.pth') 只保存加载模型参数
        """
        os.makedirs(args.save_model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'model.pth'))
        torch.save({'last_epoch': torch.tensor(epoch)}, os.path.join(args.save_model_dir, 'model.state'))
        torch.save(optimizer.state_dict(), os.path.join(args.save_model_dir, 'optimizer.pth'))


if __name__ == '__main__':
    print_arguments(args)
    train(args)
