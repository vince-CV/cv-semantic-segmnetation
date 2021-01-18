import os
from operator import itemgetter
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  
import torchvision.models as models
import torch.nn.functional as F

from tqdm.auto import tqdm  

from albumentations import Compose, Normalize, RandomCrop, HorizontalFlip, ShiftScaleRotate, HueSaturationValue
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import MultiStepLR

from trainer import Trainer, hooks
from trainer.utils import (
    setup_system,
    patch_configs,
    download_git_folder,
    get_camvid_dataset_parameters,
    draw_semantic_segmentation_batch,
    draw_semantic_segmentation_samples,
    init_semantic_segmentation_dataset,
)
from trainer.base_metric import BaseMetric
from trainer.configuration import SystemConfig, DatasetConfig, TrainerConfig, OptimizerConfig, DataloaderConfig
from trainer.matplotlib_visualizer import MatplotlibVisualizer


class SemSegDataset(Dataset):
    """ Generic Dataset class for semantic segmentation datasets.

        Arguments:
            data_path (string): Path to the dataset folder.
            images_folder (string): Name of the folder containing the images (related to the data_path).
            masks_folder (string): Name of the folder containing the masks (related to the data_path).
            num_classes (int): Number of classes in the dataset.
            transforms (callable, optional): A function/transform that inputs a sample
                and returns its transformed version.
            class_names (list, optional): Names of the classes.
            dataset_url (string, optional): url to remote repository containing the dataset.
            dataset_folder (string, optional): Folder containing the dataset (related to the git repo).

        Dataset folder structure:
            Folder containing the dataset should look like:
            - data_path
            -- images_folder
            -- masks_folder

            Names of images in the images_folder and masks_folder should be the same for same samples.
    """
    def __init__(self, data_path, images_folder, masks_folder, num_classes, transforms=None, class_names=None, dataset_url=None, dataset_folder=None):
        
        self.num_classes = num_classes
        self.transforms = transforms
        self.class_names = class_names

        if not os.path.isdir(data_path) and dataset_url is not None and dataset_folder is not None:
            download_git_folder(dataset_url, dataset_folder, data_path)

        self.dataset = init_semantic_segmentation_dataset(data_path, images_folder, masks_folder)

    def get_num_classes(self):
        return self.num_classes

    def get_class_name(self, idx):
        class_name = ""
        if self.class_names is not None and idx < len(self.num_classes):
            class_name = self.class_names[idx]
        return class_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {
            "image": cv2.imread(self.dataset[idx]["image"])[..., ::-1],
            "mask": cv2.imread(self.dataset[idx]["mask"], 0)
        }
        if self.transforms is not None:
            sample = self.transforms(**sample)
            sample["mask"] = sample["mask"].long()
        return sample


class ConfusionMatrix(BaseMetric):
    """
        Implementation of Confusion Matrix.

        Arguments:
            num_classes (int): number of evaluated classes.
            normalized (bool): if normalized is True then confusion matrix will be normalized.
    """
    def __init__(self, num_classes, normalized=False):

        self.num_classes = num_classes
        self.normalized = normalized
        self.conf = np.ndarray((num_classes, num_classes), np.int32)  # size of the confusion matrix
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def update_value(self, pred, target):                             # covert tensor into numpy array
        
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()

        valid_indices = np.where((target >= 0) & (target < self.num_classes))
        pred = pred[valid_indices]
        target = target[valid_indices]

        replace_indices = np.vstack((target.flatten(), pred.flatten())).T
        conf, _ = np.histogramdd(
            replace_indices,
            bins=(self.num_classes, self.num_classes),
            range=[(0, self.num_classes), (0, self.num_classes)]
        )

        self.conf += conf.astype(np.int32)

    def get_metric_value(self):

        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        return self.conf


class IntersectionOverUnion(BaseMetric):
    """
        Implementation of the Intersection over Union metric.

        Arguments:
            num_classes (int): number of evaluated classes.
            reduced_probs (bool): if True, then argmax was applied to the input predictions.
            normalized (bool): if normalized is True, then confusion matrix will be normalized.
            ignore_indices (int or iterable): list of ignored classes indices.
    """
    def __init__(self, num_classes, reduced_probs=False, normalized=False, ignore_indices=None):

        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, normalized=normalized)
        self.reduced_probs = reduced_probs

        if ignore_indices is None:
            self.ignore_indices = None
        elif isinstance(ignore_indices, int):
            self.ignore_indices = (ignore_indices, )
        else:
            try:
                self.ignore_indices = tuple(ignore_indices)
            except TypeError:
                raise ValueError("'ignore_indices' must be an int or iterable")

    def reset(self):
        self.conf_matrix.reset()

    def update_value(self, pred, target):

        if not self.reduced_probs:
            pred = pred.argmax(dim=1)
        self.conf_matrix.update_value(pred, target)

    def get_metric_value(self):

        conf_matrix = self.conf_matrix.get_metric_value()

        if self.ignore_indices is not None:
            conf_matrix[:, self.ignore_indices] = 0
            conf_matrix[self.ignore_indices, :] = 0

        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        if self.ignore_indices is not None:
            iou_valid_cls = np.delete(iou, self.ignore_indices)
            miou = np.nanmean(iou_valid_cls)
        else:
            miou = np.nanmean(iou)
        return {"mean_iou": miou, "iou": iou}


"""test_dataset = SemSegDataset(**get_camvid_dataset_parameters(
        data_path="data",
        dataset_type="test"
    ))
draw_semantic_segmentation_samples(test_dataset, n_samples=2)
# create intersection over union instance
metric = IntersectionOverUnion(num_classes=test_dataset.get_num_classes(), reduced_probs=True)
# add samples from test dataset and update metrics value
for sample in tqdm(test_dataset, ascii=False):
    masks = sample["mask"]
    metric.update_value(masks, masks)
# get the mean iou value
values = metric.get_metric_value()
print(values['mean_iou'])
print(values['iou'])"""


class ResNetEncoder(nn.Module):

    """ ResNet encoder.

        Arguments:
            resnet_type (string): type of resnet (all resnet network which exist in torchvision).
            pretrained (bool): if pretrained == True, ImageNet weights will load.
    """
    def __init__(self, resnet_type="resnet18", pretrained=True):
        super().__init__()
        self.module = getattr(models, resnet_type)(pretrained=pretrained)


    def get_channels_out(self):
        channels_out = []
        for layer in [getattr(self.module, "layer{}".format(i)) for i in range(1, 5)]:
            channels_out.append(self._get_block_size(layer))
        return channels_out[::-1]

    def forward(self, x):
        x = self.module.conv1(x)
        x = self.module.bn1(x)
        x = self.module.relu(x)
        x = self.module.maxpool(x)

        l1_output = self.module.layer1(x)           # num of out chns = 64
        l2_output = self.module.layer2(l1_output)   # num of out chns = 128
        l3_output = self.module.layer3(l2_output)   # num of out chns = 256
        l4_output = self.module.layer4(l3_output)   # num of out chns = 512

        return l1_output, l2_output, l3_output, l4_output

    @staticmethod
    def _get_block_size(module):
        return list(module[-1].modules())[-2].weight.size()[0]


class LateralConnection(nn.Module):
    """
        Lateral connection.

        Arguments:
            channels_in (int): number of input channels.
            channels_out (int): number of output channels.
    """
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.proj = nn.Conv2d(channels_in, channels_out, kernel_size=1)

    def forward(self, prev, cur):
        up = F.interpolate(prev, cur.size()[-2:], mode="nearest")
        proj = self.proj(cur)
        return proj + up


class FPNDecoder(nn.Module):
    """
        Feature Pyramid Decoder.
            Aggregate all feature pyramid layers using lateral connections.

        Arguments:
            channels_in (list): list of input channels for each feature pyramid layer.
            channels_out (int): number of output channels.
    """
    def __init__(self, channels_in, channels_out=256):
        super().__init__()

        self.module = nn.ModuleList()
        self.module.append(nn.Conv2d(channels_in[0], channels_out, kernel_size=1))

        for i in range(1, len(channels_in)):
            self.module.append(LateralConnection(channels_in[i], channels_out))

    def forward(self, x):

        output = [self.module[0](x[0])]
        for i in range(1, len(x)):
            output.append(self.module[i](output[i - 1], x[i]))
        return output


class SemanticSegmentation(nn.Module):
    """
        Semantic Segmentation model using Feature Pyramid Network.

        Arguments:
            num_classes (int): number of classes.
            encoder_type (class): type of encoder network.
            channels_out (int): number of channels of the output features.
            final_upsample (bool): if final_upsample is True then final prediction will be upsampled to the original resolution.
    """
    def __init__(self, num_classes, encoder_type=ResNetEncoder, channels_out=128, final_upsample=False):
        super().__init__()

        self.final_upsample = final_upsample
        self.encoder = encoder_type()
        self.decoder = FPNDecoder(self.encoder.get_channels_out(), channels_out=channels_out)

        self.classifier = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, num_classes, kernel_size=1),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder[::-1])
        classifier = self.classifier(decoder[-1])

        if self.final_upsample:
            classifier = F.interpolate(classifier, x.size()[-2:], mode="bilinear", align_corners=False)

        return classifier


class Experiment:
    def __init__(self,
        system_config:      SystemConfig  = SystemConfig(),
        dataset_config:     DatasetConfig = DatasetConfig(),
        dataloader_config:  DataloaderConfig = DataloaderConfig(),
        optimizer_config:   OptimizerConfig  = OptimizerConfig(),
    ):

        self.system_config = system_config
        setup_system(system_config)

        self.loader_train = DataLoader(
            SemSegDataset(
                **get_camvid_dataset_parameters(
                    data_path=dataset_config.root_dir,
                    dataset_type="train",
                    transforms=Compose([
                        HorizontalFlip(),
                        ShiftScaleRotate(
                            shift_limit=0.0625,
                            scale_limit=0.50,
                            rotate_limit=45,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=11,
                            p=.75
                        ),
                        HueSaturationValue(),
                        RandomCrop(height=352, width=480),
                        Normalize(),
                        ToTensorV2()
                    ])
                )
            ),
            batch_size = dataloader_config.batch_size,
            shuffle = True,
            num_workers = dataloader_config.num_workers,
            pin_memory = True
        )


        self.loader_test = DataLoader(
            SemSegDataset(
                **get_camvid_dataset_parameters(
                    data_path=dataset_config.root_dir,
                    dataset_type="test",
                    transforms=Compose([Normalize(), ToTensorV2()])
                )
            ),
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            num_workers=dataloader_config.num_workers,
            pin_memory=True
        )


        self.model = SemanticSegmentation(
            num_classes=self.loader_test.dataset.get_num_classes(), final_upsample=True
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.loader_test.dataset.get_num_classes()
        )

        self.metric_fn = IntersectionOverUnion(
            num_classes=self.loader_test.dataset.get_num_classes(), reduced_probs=False
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay
        )

        self.lr_scheduler = MultiStepLR(
            self.optimizer, milestones=optimizer_config.lr_step_milestones, gamma=optimizer_config.lr_gamma
        )

        self.visualizer = MatplotlibVisualizer()


    def run(self, trainer_config: TrainerConfig) -> dict:

        setup_system(self.system_config)

        device = torch.device(trainer_config.device)
 
        self.model = self.model.to(device)
        self.loss_fn = self.loss_fn.to(device)


        model_trainer = Trainer(
            model=self.model,
            loader_train=self.loader_train,
            loader_test=self.loader_test,
            loss_fn=self.loss_fn,
            metric_fn=self.metric_fn,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            data_getter=itemgetter("image"),
            target_getter=itemgetter("mask"),
            stage_progress=trainer_config.progress_bar,
            get_key_metric=itemgetter("mean_iou"),
            visualizer=self.visualizer,
            model_saving_frequency=trainer_config.model_saving_frequency,
            save_dir=trainer_config.model_dir
        )


        model_trainer.register_hook("end_epoch", hooks.end_epoch_hook_semseg)
        # run the training
        self.metrics = model_trainer.fit(trainer_config.epoch_num)
        return self.metrics



dataloader_config, trainer_config = patch_configs(epoch_num_to_set=30, batch_size_to_set=16)
optimizer_config = OptimizerConfig(learning_rate=1e-3, lr_step_milestones =[], weight_decay=4e-5)
experiment = Experiment(dataloader_config=dataloader_config, optimizer_config=optimizer_config)
metrics = experiment.run(trainer_config)

plt.figure()
# plot test and train loss
plt.plot(metrics["test_loss"], label="test")
plt.plot(metrics["train_loss"], label="train")
plt.legend()
plt.show()


plt.figure()
# plot mean iou metric for all classes
plt.plot([metric['mean_iou'] for metric in metrics["test_metric"]], label="miou")
plt.legend()
plt.show()