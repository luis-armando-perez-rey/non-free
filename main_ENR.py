from utils.parse_args import get_args
import torch.utils.data
from torch import save
import sys
# Import datasets
from datasets.equiv_dset import *
from models.models_nn import *
from utils.nn_utils import *
from models.losses import ReconstructionLoss
from models.distributions import MixtureDistribution, get_prior, get_z_values
from ENR.models_ENR import *
from ENR.config import ENR_CONFIG
from ENR.transforms3d.conversions import rotation_matrix_z

# region PARSE ARGUMENTS
parser = get_args()
args = parser.parse_args()
if args.neptune_user != "":
    from utils.neptune_utils import initialize_neptune_run, save_sys_id

    run = initialize_neptune_run(args.neptune_user, "non-free")
    run["parameters"] = vars(args)
else:
    run = None
print(args)

# endregion

# region TORCH SETUP
# Print set up torch device, empty cache and set random seed
torch.cuda.empty_cache()
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# endregion

# region PATHS
# Paths were the models will be saved
model_path = os.path.join(args.checkpoints_dir, args.model_name)
make_dir(model_path)
decoder_file = os.path.join(model_path, "decoder.pt")
# Paths for saving the images
figures_dir = os.path.join(model_path, 'figures')
make_dir(figures_dir)
model_file = os.path.join(model_path, 'model.pt')
meta_file = os.path.join(model_path, 'metadata.pkl')
# Neptune txt file
if run is not None:
    neptune_id_path = os.path.join(model_path, 'neptune.txt')
    save_sys_id(run, neptune_id_path)

# endregion

# region SAVE METADATA
# Save the arguments
pickle.dump({'args': args}, open(meta_file, 'wb'))
# endregion

# region SET DATASET
if args.dataset == 'platonics':
    dset = PlatonicMerged(N=30000, data_dir=args.data_dir)
else:
    print(f"Loading dataset {args.dataset} with dataset name {args.dataset_name}")


    if args.dataset != "symmetric_solids":
        dset = EquivDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name,
                            max_data_per_dataset=args.ndatapairs, so3_matrices=True)
        dset_eval = EvalDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name)
        eval_images = torch.FloatTensor(dset_eval.data.reshape(-1, dset_eval.data.shape[-1]))
        stabilizers = dset_eval.stabs.reshape((-1))
    else:
        dset = EquivDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name,
                            max_data_per_dataset=args.ndatapairs)

# Setup torch dataset
dset, dset_test = torch.utils.data.random_split(dset, [len(dset) - int(len(dset) / 10), int(len(dset) / 10)])
train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_test, batch_size=args.batch_size, shuffle=True)
print("# train set:", len(dset))
print("# test set:", len(dset_test))


# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]
# endregion


model = NeuralRenderer(**ENR_CONFIG).to(device)

rec_loss_function = ReconstructionLoss(args.reconstruction_loss)
optimizer = get_optimizer(args.optimizer, args.lr, model.parameters())

errors = []


def equivariance_loss(z_transformed, z_next):
    loss = ((z_transformed - z_next) ** 2).mean()
    return loss


def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    equiv_loss = 0
    global_step = len(data_loader) * epoch

    # Initialize the losses that will be used for logging with Neptune
    total_batches = len(data_loader)

    for batch_idx, (image, img_next, action) in enumerate(data_loader):
        batch_size = image.shape[0]
        global_step += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            model.train()

        elif mode == 'val':
            model.eval()

        image = image.to(device)

        img_next = img_next.to(device)
        action = action.to(device).squeeze(1)

        x_rec, _ = model(image, action)
        encoded_image = model.encode(image)
        encoded_image_next = model.encode(image)
        encoded_image_transformed = model.act(encoded_image, action)


        equiv_loss_batch = equivariance_loss(encoded_image_transformed, encoded_image_next).item()
        equiv_loss += equiv_loss_batch

        # equiv_loss_function(p_next, p_pred)
        # print(encoded_image.shape, encoded_image_next.shape, encoded_image_transformed.shape)

        loss = rec_loss_function(x_rec, img_next).mean()

        if run is not None:
            run[mode + "/batch/loss"].log(loss)
            run[mode + "/batch/equivariance_loss"].log(equiv_loss_batch)

        print(
            f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} "
            f"Equivariance Loss: {equiv_loss_batch:.3f}")

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    mu_loss /= total_batches
    equiv_loss /= total_batches

    if mode == 'val':
        errors.append(mu_loss)
        np.save(f'{model_path}/errors_val.npy', errors)
        if run is not None:
            run[mode + "/epoch/val_loss"].log(mu_loss)
            run[mode + "/epoch/equiv_loss"].log(equiv_loss)
        if (epoch % args.save_interval) == 0:
            save(model, model_file)
            print("Model saved", model_file)

    # Write to standard output. Allows printing to file progress when using HPC
    sys.stdout.flush()

    # Get each loss component to be logged by Neptune
    epoch_loss = mu_loss
    if run is not None:
        run[mode + "/epoch/loss"].log(epoch_loss)


if __name__ == "__main__":
    for epoch_ in range(1, args.epochs + 1):
        train(epoch_, train_loader, 'train')
        with torch.no_grad():
            train(epoch_, val_loader, 'val')
    if run is not None:
        run.stop()
