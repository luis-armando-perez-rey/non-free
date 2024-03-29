import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import resize
from typing import Tuple, Optional, List
import torchvision
from torch.utils.data import DataLoader


def process_pil_image(img):
    img = np.expand_dims(np.array(img), axis = 0)
    img = img / 255.0
    return img

def generate_training_data(dataset_folder, dataset_name,
                           examples_per_digit: int = 1):
    # train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                                                   ])
    torch_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    print("Total train digits: ", len(torch_dataset))

    # Dictionary in case we want to convert the digit labels to number of stabilizers
    stabilizer_dict = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 2, 9: 2}

    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for num_digit in range(len(torch_dataset)):
        for num_example in range(examples_per_digit):
            image, target = torch_dataset.__getitem__(num_digit)
            print(
                "Generating image with digit number {}, num example {}".format(target, num_example))

            angle1 = 360 * np.random.random()
            angle2 = 360 * np.random.random()
            angle = 2 * np.pi * (angle2 - angle1) / 360
            img1 = process_pil_image(image.rotate(angle1))
            img2 = process_pil_image(image.rotate(angle2))
            equiv_data.append([img1, img2])
            equiv_lbls.append(angle)
            equiv_stabilizers.append(target)

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_training_data_stochastic(dataset_folder, dataset_name,
                           examples_per_digit: int = 1):
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      ])
    torch_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)



    # Dictionary in case we want to convert the digit labels to number of stabilizers
    stabilizer_dict = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 1, 8: 2, 9: 2}

    equiv_data = []
    equiv_lbls = []
    equiv_stabilizers = []
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    batch_size = 1
    for num_digit in range(10):


        # Select the images from a certain digit

        dataset = list(filter(lambda i: i[1] == num_digit, torch_dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iter_dataset = iter(loader)

        for num_example in range(examples_per_digit):
            image1 = next(iter_dataset)[0][0]
            image2 = next(iter_dataset)[0][0]

            print(
                "Generating image with digit number {}, num example {}".format(num_digit, num_example))

            angle1 = 360 * np.random.random()
            angle2 = 360 * np.random.random()
            angle = np.pi * (angle2 - angle1) / 180
            image1 = torchvision.transforms.functional.rotate(image1, angle1)
            image2 = torchvision.transforms.functional.rotate(image2, angle2)
            # print(image1.shape, image1.unique())

            equiv_data.append([image1.numpy(), image2.numpy()])
            equiv_lbls.append(angle)
            equiv_stabilizers.append(num_digit)

    equiv_data = np.array(equiv_data)
    equiv_lbls = np.array(equiv_lbls)
    equiv_stabilizers = np.array(equiv_stabilizers)
    print("Equiv data shape", equiv_data.shape)
    print("Equiv lbls shape", equiv_lbls.shape)
    print("Equiv stabilizers shape", equiv_stabilizers.shape)

    np.save(os.path.join(dataset_folder, dataset_name + '_data.npy'), equiv_data)
    np.save(os.path.join(dataset_folder, dataset_name + '_lbls.npy'), equiv_lbls)
    np.save(os.path.join(dataset_folder, dataset_name + '_stabilizers.npy'), equiv_stabilizers)


def generate_eval_data(dataset_folder, dataset_name, total_digits: int = 100,
                       total_rotations: int = 36):
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                      ])
    torch_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    # Regular dataset
    images = []
    stabilizers = []
    labels = []
    angles = np.linspace(0, 1, total_rotations) * 360
    print("Total eval digits: ", len(torch_dataset))
    total_digits = np.amin([total_digits, len(torch_dataset)])
    for num_digit in range(total_digits):
        images_per_object = []
        labels_per_object = []
        for num_angle, angle in enumerate(angles):
            image, target = torch_dataset.__getitem__(num_digit)
            print(
                "Generating image with digit number {}, num example {}".format(target, num_angle))
            img = process_pil_image(image.rotate(angle))
            images_per_object.append(img)
            labels_per_object.append(angle * np.pi / 180)

        stabilizers.append([target] * total_rotations)
        images.append(images_per_object)
        labels.append(labels_per_object)

    images = np.array(images)
    stabilizers = np.array(stabilizers)
    labels = np.array(labels)
    print("Images shape", images.shape)
    print("Stabilizers shape", stabilizers.shape)
    print("Labels shape", labels.shape)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_data.npy'), images)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_lbls.npy'), labels)
    np.save(os.path.join(dataset_folder, dataset_name + '_eval_stabilizers.npy'), stabilizers)

if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      # torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                      ])
    torch_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    # image = torchvision.transforms.functional.rotate(torch_dataset[0], 45)

    batch_size = 5
    dataset = list(filter(lambda i: i[1] == 0, torch_dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Total train digits: ", len(loader))


    iter_dataset = iter(loader)
    images, labels = next(iter_dataset)
    fig, axes = plt.subplots(1,batch_size)
    for num_image, image in enumerate(images):
        axes[num_image].imshow(image.squeeze().numpy())
        print(np.unique(image.numpy()), image.shape)
    plt.show()
    iter_dataset = iter(loader)
    images, labels = next(iter_dataset)
    fig, axes = plt.subplots(1, batch_size)
    for num_image, image in enumerate(images):
        axes[num_image].imshow(image.squeeze().numpy())
    plt.show()
    # train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                                                   ])
    # torch_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    # image, label = torch_dataset[0]
    # image = np.array(image.rotate(45))
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    # torch_dataset[0][0].show()
    #
    # train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                   # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #                                                   ])
    # torch_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=train_transform)
    # image, target = torch_dataset.__getitem__(0)
    # img = torchvision.transforms.functional.rotate(image, 45).numpy()
    # plt.imshow(img[0])
    # print(img.shape)
    # plt.show()



