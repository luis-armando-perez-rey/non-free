from datasets.equiv_dset import EquivDatasetStabs, EvalDataset, PlatonicMerged


def get_dataset(data_dir, dataset, dataset_name, so3_matrices=False):
    if dataset == 'square':
        dset = EquivDatasetStabs(f'{data_dir}/square/', list_dataset_names=dataset_name, so3_matrices=so3_matrices)
        dset_eval = None

    elif dataset == 'platonics':
        dset = PlatonicMerged(N=30000, data_dir=data_dir)
        dset_eval = None

    elif dataset == "symmetric_solids":
        dset = EquivDatasetStabs(f'{data_dir}/symmetric_solids/', list_dataset_names=dataset_name)
        dset_eval = dset
    else:
        dset = EquivDatasetStabs(f'{data_dir}/{dataset}/', list_dataset_names=dataset_name, so3_matrices=so3_matrices)
        dset_eval = EvalDataset(f'{data_dir}/{dataset}/', list_dataset_names=dataset_name)
    return dset, dset_eval

