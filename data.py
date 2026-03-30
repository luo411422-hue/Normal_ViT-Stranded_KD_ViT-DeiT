import torch
import torchvision
import torchvision.transforms as transforms

import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset

"""CIFAR - 10 dataset loading code, you can replace it with your own dataset if you want, but make sure to change the input channels of the patch embedding layer accordingly (default is 3"""
class CIFAR10():
    def get_loader():
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                            0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                            0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=r'E:\Python\datasets\CIFAR-10', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=r'E:\Python\datasets\CIFAR-10', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        return trainloader, testloader


class CIFAR100():
    def get_loader():
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(
            root=r'E:\Python\datasets\CIFAR-100', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root=r'E:\Python\datasets\CIFAR-100', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)
        return trainloader, testloader




class CassavaLeafDiseaseDataset(Dataset):#数据集类，继承自 PyTorch 的 Dataset 类，用于加载 Cassava 叶病数据集。
    def __init__(self, image_dir, samples, transform=None):
        self.image_dir = Path(image_dir)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
       image_name, label = self.samples[index]               # 例如 ("100002.jpg", 1)
       image_path = self.image_dir / image_name              # 拼接路径：train_images/100002.jpg
       image = Image.open(image_path).convert("RGB")        # 打开图像
       #print(image.size) # 输出图像尺寸，供调试使用
       if self.transform is not None:
            image = self.transform(image)

       return image, label #返回图像数据和对应的标签，供数据加载器使用。


class CassavaData: #数据处理类
    DEFAULT_ZIP_PATH = Path(r"E:\paper\transformer-master\cassava-leaf-disease-classification (1).zip")
    DEFAULT_EXTRACT_DIR = Path(r"E:\paper\transformer-master\cassava_leaf_disease")

    @classmethod
    def prepare_dataset(cls, zip_path=None, extract_dir=None):  #准备数据集，解压缩并返回数据集目录路径。
        zip_path = Path(zip_path) if zip_path else cls.DEFAULT_ZIP_PATH
        extract_dir = Path(extract_dir) if extract_dir else cls.DEFAULT_EXTRACT_DIR

        train_csv = extract_dir / "train.csv"
        train_images_dir = extract_dir / "train_images"

        if train_csv.exists() and train_images_dir.exists():
            return extract_dir

        if not zip_path.exists():
            raise FileNotFoundError(
                f"Cassava dataset zip not found: {zip_path}. "
                "Please place the Kaggle zip in the project root or pass a custom path."
            )

        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)

        return extract_dir

    @staticmethod
    def _read_train_samples(dataset_dir): #读取训练样本，从 train.csv 文件中提取图像文件名和对应的标签，并返回一个列表。
        samples = []
        train_csv_path = Path(dataset_dir) / "train.csv"
        with train_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                samples.append((row["image_id"], int(row["label"]))) #将图像文件名和标签添加到 samples 列表中。
        return samples

    @staticmethod
    def _stratified_split(samples, val_ratio=0.2, seed=42):#按照标签进行分层划分，将样本按照标签分组，然后在每个标签组内随机划分训练集和验证集，确保每个标签在训练集和验证集中都有代表性。 
        grouped_samples = defaultdict(list) #使用 defaultdict 将样本按照标签进行分组，键为标签，值为对应标签的样本列表。
        for sample in samples:
            grouped_samples[sample[1]].append(sample) #遍历样本列表，将每个样本根据其标签添加到对应的标签组中。

        rng = random.Random(seed) #使用随机数生成器设置随机种子，以确保结果可复现。
        train_samples = []#创建一个空列表 train_samples 用于存储划分后的训练样本。
        val_samples = []#创建一个空列表 val_samples 用于存储划分后的验证样本。

        for label_samples in grouped_samples.values():#遍历每个标签组的样本列表，对每个标签组内的样本进行随机划分。
            rng.shuffle(label_samples)#对当前标签组内的样本进行随机打乱，以确保划分的随机性。
            val_count = max(1, int(len(label_samples) * val_ratio)) #计算验证集的样本数量，至少保证每个标签至少有一个样本被划分到验证集中。
            if val_count >= len(label_samples):
                val_count = len(label_samples) - 1

            val_samples.extend(label_samples[:val_count]) #将当前标签组内的前 val_count 个样本添加到验证集列表 val_samples 中。
            train_samples.extend(label_samples[val_count:])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        return train_samples, val_samples #返回划分后的训练样本列表和验证样本列表。

    @staticmethod
    def _build_transforms(img_size): #构建数据增强和预处理的变换管道，返回训练集和验证集的变换对象。
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),#首先将图像调整为比目标大小更大的尺寸，以便后续的随机裁剪能够产生更多样化的训练样本。
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),#随机裁剪图像为目标大小 img_size，并且裁剪的区域大小在原图的 80% 到 100% 之间，这有助于模型学习到不同尺度的特征。
            transforms.RandomHorizontalFlip(),#随机水平翻转图像，增加数据的多样性，帮助模型更好地泛化。
            transforms.RandomRotation(15),#随机旋转图像，增加数据的多样性，帮助模型更好地泛化。
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),#随机调整图像的亮度、对比度、饱和度和色调，增加数据的多样性，帮助模型更好地泛化。
            transforms.ToTensor(),#将图像转换为 PyTorch 的张量格式，并且将像素值归一化到 [0, 1] 范围内。
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#对图像进行标准化处理，使用 ImageNet 数据集的均值和标准差进行归一化，这有助于模型更快地收敛。
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),#将图像调整为目标大小 img_size，确保验证集的图像尺寸与训练集一致。
            transforms.ToTensor(),#将图像转换为 PyTorch 的张量格式，并且将像素值归一化到 [0, 1] 范围内。
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#对图像进行标准化处理，使用 ImageNet 数据集的均值和标准差进行归一化，这有助于模型更快地收敛。
        ])

        return train_transform, eval_transform # 返回训练集的变换对象和验证集的变换对象，供后续的数据加载器使用。

    @classmethod
    def get_loaders( #定义一个类方法 get_loaders，用于获取训练集和验证集的数据加载器。
        cls,#类方法的第一个参数 cls 代表类本身，可以通过 cls 来调用类中的其他方法或属性。
        batch_size=128,
        img_size=224,
        val_ratio=0.2,#验证集的比例
        seed=42,
        num_workers=0,#数据加载的工作线程数，设置为 0 表示在主线程中加载数据，适用于 Windows 系统或调试阶段。
        zip_path=None,#数据集的 zip 文件路径，如果未提供则使用默认路径。
        extract_dir=None,
    ):
        dataset_dir = cls.prepare_dataset(zip_path=zip_path, extract_dir=extract_dir)#准备数据集，返回数据集目录路径。
        all_samples = cls._read_train_samples(dataset_dir)#读取训练集样本，返回样本列表。
        train_samples, val_samples = cls._stratified_split( #按照标签进行分层划分，返回训练样本列表和验证样本列表。
            all_samples, val_ratio=val_ratio, seed=seed
        )
        train_transform, eval_transform = cls._build_transforms(img_size) #构建训练集和验证集的变换对象。

        train_dataset = CassavaLeafDiseaseDataset( #创建训练集数据集对象，传入图像目录路径、训练样本列表和训练变换对象。
            image_dir=Path(dataset_dir) / "train_images",
            samples=train_samples,
            transform=train_transform,
        )
        val_dataset = CassavaLeafDiseaseDataset(
            image_dir=Path(dataset_dir) / "train_images",
            samples=val_samples,
            transform=eval_transform,
        )

        train_loader = DataLoader( #创建训练集数据加载器，传入训练集数据集对象、批大小、是否打乱数据、工作线程数和是否使用 CUDA 加速的数据加载。
            train_dataset,
            batch_size=batch_size,#每个批次加载的样本数量，较大的批次可以提高训练效率，但也会增加内存使用。
            shuffle=True,
            num_workers=num_workers,#数据加载的工作线程数，设置为 0 表示在主线程中加载数据，适用于 Windows 系统或调试阶段。
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return train_loader, val_loader #返回训练集和验证集的数据加载器，供模型训练和评估使用。

    @classmethod
    def get_label_map(cls, zip_path=None, extract_dir=None):#获取标签映射，从数据集目录中的 label_num_to_disease_map.json 文件中读取标签映射关系，并返回一个字典。
        dataset_dir = cls.prepare_dataset(zip_path=zip_path, extract_dir=extract_dir) #准备数据集，返回数据集目录路径。
        label_map_path = Path(dataset_dir) / "label_num_to_disease_map.json" #构建标签映射文件的完整路径，通过将数据集目录路径 dataset_dir 与文件名 label_num_to_disease_map.json 进行拼接得到。
        with label_map_path.open("r", encoding="utf-8") as json_file:#使用 UTF-8 编码打开标签映射文件，并将其作为 json_file 对象进行读取。
            return json.load(json_file) #使用 json 模块的 load 函数从 json_file 对象中读取数据，并将其解析为一个 Python 字典对象，返回该字典作为标签映射关系。
        
# def main():
#     train_loader, val_loader = CassavaData.get_loaders()
#     for i in range(10): #从训练集数据加载器的 dataset 属性中获取第一个样本，返回图像数据和对应的标签。
#         image, label = train_loader.dataset[i]
#         label_map = CassavaData.get_label_map() #获取标签映射关系，返回一个字典，其中键为标签编号，值为对应的疾病名称。
#         print(f"sql: {i}, Image size: {image.size()}, Label: {label}")
#         print(f"Label map: {label_map.get(str(label), 'Unknown')}")

# if __name__ == "__main__":
#     main()

