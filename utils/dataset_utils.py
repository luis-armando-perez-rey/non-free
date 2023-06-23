import numpy as np
import torch
from datasets.equiv_dset import EquivDatasetStabs, EvalDataset, PlatonicMerged, FactorDataset, EquivDataset
from dataset_generation.modelnet_efficient import ModelNetDataset, ModelNetDatasetComplete, ModelNetEvalDataset


def get_loading_parameters(data_dir: str, dataset, dataset_name, so3_matrices: bool = False):
    loading_parameters = []
    if dataset.startswith("modelnet_efficient"):
        train_data_parameters = dict(render_folder="/data/volume_2/data/active_views",
                                     split="train",
                                     object_type_list=dataset_name,
                                     examples_per_object=12,
                                     use_random_initial=True,
                                     total_views=360,
                                     fixed_number_views=12,
                                     shuffle_available_views=True,
                                     use_random_choice=False)
        if dataset == "modelnet_efficient224":
            train_data_parameters["resolution"] = 224

        elif dataset == "modelnet_efficient_test":
            train_data_parameters["split"] = "test"
        elif dataset == "modelnet_efficient_test_initial0":
            train_data_parameters["split"] = "test"
            train_data_parameters["use_random_initial"] = False
        elif dataset == "modelnet_efficient_init0":
            train_data_parameters["split"] = "test"
            train_data_parameters["use_random_initial"] = False
        elif dataset == "modelnet_efficient_test_initialrnd":
            train_data_parameters["split"] = "test"
            train_data_parameters["use_random_initial"] = True
        elif dataset == "modelnet_efficient_single":
            train_data_parameters["object_ids"] = [list(np.arange(15))] * len(dataset_name)




        loading_parameters.append(train_data_parameters)
        eval_data_parameters = train_data_parameters.copy()
        eval_data_parameters.pop("shuffle_available_views")
        eval_data_parameters["seed"] = 70
        loading_parameters.append(eval_data_parameters)
    elif dataset.startswith("shrec21shape"):
        train_data_parameters = dict(render_folder="./data/shrec21shape",
                                     split="train",
                                     object_type_list=dataset_name,
                                     examples_per_object=12,
                                     use_random_initial=True,
                                     total_views=12,
                                     fixed_number_views=12,
                                     shuffle_available_views=True,
                                     use_random_choice=False)
        if dataset == "shrec21shape224":
            train_data_parameters["resolution"] = 224

        elif dataset == "shrec21shape_test":
            train_data_parameters["split"] = "test"
        loading_parameters.append(train_data_parameters)
        eval_data_parameters = train_data_parameters.copy()
        eval_data_parameters.pop("shuffle_available_views")
        eval_data_parameters["seed"] = 70
        loading_parameters.append(eval_data_parameters)


    return loading_parameters


def get_dataset(data_dir, dataset, dataset_name, so3_matrices=False):
    if dataset == 'square':
        dset = EquivDatasetStabs(f'{data_dir}/square/', list_dataset_names=dataset_name, so3_matrices=so3_matrices)
        dset_eval = None

    elif dataset == 'platonics':
        dset = PlatonicMerged(N=30000, data_dir=data_dir)
        dset_eval = None

    elif dataset == "symmetric_solids":
        dset = EquivDatasetStabs(f'{data_dir}/symmetric_solids/', list_dataset_names=dataset_name)
        eval_dset_names = [dataset_name + "_eval" for dataset_name in dataset_name]
        dset_eval = EquivDatasetStabs(f'{data_dir}/symmetric_solids/', list_dataset_names=eval_dset_names)
    elif dataset == "modelnetso3":
        dset = EquivDatasetStabs(f'{data_dir}/modelnetso3/', list_dataset_names=dataset_name)
        dset_eval = dset
    elif dataset == "modelnet_regular" or dataset == "modelnet_regular_pairs" or dataset == "modelnet_regular_pairs0" or dataset == "modelnet_regular0":
        dset = FactorDataset(f'{data_dir}/{dataset}/', list_dataset_names=dataset_name, factor_list=["orbit", "class"])
        eval_dset_names = [dataset_name + "_val" for dataset_name in dataset_name]
        dset_eval = FactorDataset(f'{data_dir}/{dataset}/', list_dataset_names=eval_dset_names,
                                  factor_list=["orbit", "class"])
    elif dataset == "modelnet_regular_pairs_test":
        eval_dset_names = [dataset_name + "_eval" for dataset_name in dataset_name]
        dset = FactorDataset(f'{data_dir}/{dataset}/', list_dataset_names=eval_dset_names,
                             factor_list=["orbit", "class"])
        dset_eval = None
    elif dataset.startswith("modelnet_efficient"):
        loading_parameters = get_loading_parameters(data_dir, dataset, dataset_name, so3_matrices)
        dset = ModelNetDatasetComplete(**loading_parameters[0])
        dset_eval = ModelNetEvalDataset(**loading_parameters[1])
    elif dataset.startswith("shrec21shape"):
        loading_parameters = get_loading_parameters(data_dir, dataset, dataset_name, so3_matrices)
        dset = ModelNetDatasetComplete(**loading_parameters[0])
        dset_eval = ModelNetEvalDataset(**loading_parameters[1])



    else:
        dset = EquivDatasetStabs(f'{data_dir}/{dataset}/', list_dataset_names=dataset_name, so3_matrices=so3_matrices)
        dset_eval = EvalDataset(f'{data_dir}/{dataset}/', list_dataset_names=dataset_name)
    return dset, dset_eval


def get_unique_data(data_array: np.array, identifiers: np.array, num_unique_examples: int = 1) -> np.array:
    """
    Get unique examples from the dataset based on some identifiers
    :param data_array: dataset of type EquivDataset or subclass
    :param identifiers: array of identifiers
    :param num_unique_examples: number of unique examples to return
    :return:
    """
    assert len(identifiers) == len(data_array), "Identifiers and data array must be the same length"
    unique_data = []
    unique_identifiers = np.unique(identifiers)
    for unique in unique_identifiers:
        if identifiers.ndim == 2:
            unique_data.append(data_array[np.product(identifiers == unique, axis=-1)][0][:num_unique_examples])
        else:
            unique_data.append(data_array[identifiers == unique][0][:num_unique_examples])
    unique_data = np.concatenate(unique_data, axis=0)
    if unique_data.ndim == 3:
        unique_data = np.expand_dims(unique_data, axis=0)
    return unique_data


def get_data_from_dataloader(dataloader, data_index: int = 0):
    """
    Get the embeddings of the evaluation dataset
    :param dataloader: dataloader
    :param data_index: index of the data to get
    :return:
    """
    # region GET DATA
    data = []
    for num_batch, batch in enumerate(dataloader):
        data.append(batch[data_index].detach())
    data = torch.cat(data, dim=0)
    print("Data shape", data.shape)
    return data
