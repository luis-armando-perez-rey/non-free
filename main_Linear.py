from utils.parse_args import get_args
import torch.utils.data
from torch import save
import sys
# Import datasets
from datasets.equiv_dset import *
from models.resnet import *
from models.models_nn import *
from utils.nn_utils import *
from utils.plotting_utils import save_embeddings_on_circle
from models.losses import EquivarianceLoss, ReconstructionLoss, IdentityLoss, estimate_entropy, matrix_dist
from models.distributions import MixtureDistribution, get_prior, get_z_values

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
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
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
    dset = EquivDataset(f'{args.data_dir}/{args.dataset}/', list_dataset_names=args.dataset_name,
                        max_data_per_dataset=args.ndatapairs)

    dset_val = EquivDataset(f'{args.data_dir}/{args.dataset}/',
                            list_dataset_names=[dataset_name + "_val" for dataset_name in args.dataset_name],
                            max_data_per_dataset=-1)

# Setup torch dataset
# dset, dset_val = torch.utils.data.random_split(dset, [len(dset) - int(len(dset) / 10), int(len(dset) / 10)])
train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size, shuffle=True)

print("# train set:", len(dset))
print("# test set:", len(dset_val))

# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]
# endregion

print("Using ResNet18")
model = ResNet18Enc(z_dim=args.latent_dim, nc=img_shape[0]).to(device)

dec = None

optimizer = get_optimizer(args.optimizer, args.lr, model.parameters())
errors = []
equiv_errors = []
errors_hitrate = []



def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    mu_hit_rate = 0
    mu_equiv_loss = 0

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
        action = action.to(device)

        # z_mean and z_mean_next have shape (batch_size, n, latent_dim)
        z_mean = model(image)
        z_mean_next = model(img_next)


        # Rotate the embeddings in Z_G of the first image by the action
        if args.latent_dim == 2:
            action = torch.flatten(action)
            rot = make_rotation_matrix(action).squeeze(1)
        else:
            rot = action
        z_mean_pred = (rot @ z_mean.unsqueeze(-1)).squeeze(-1)

        # Calculate equivariance loss
        loss_equiv = ((z_mean_next - z_mean_pred)**2).sum(-1).mean()
        losses = [loss_equiv]
        mu_equiv_loss += loss_equiv.item()
        if run is not None:
            run[mode + "/batch/loss_equiv"].log(loss_equiv)

        # Hinge regularization
        z_mean_rand = z_mean[torch.randperm(len(z_mean))]
        distance = ((z_mean_rand - z_mean)**2).sum(-1)
        hinge_reg = 0.001*torch.max(torch.zeros_like(distance).to(device), 1.*torch.ones_like(distance).to(device) - distance).mean()
        losses.append(hinge_reg)



        dist_matrix = ((z_mean_next.unsqueeze(0) - z_mean_pred.unsqueeze(1)) ** 2).sum(-1)
        hitrate = hitRate_generic(dist_matrix, image.shape[0])
        mu_hit_rate += hitrate.item()
        # endregion




        # Sum all losses
        if mode == "train":
            loss = sum(losses)
        elif mode == "val":
            loss = loss_equiv / args.weightequivariance
        else:
            loss = None
        if run is not None:
            run[mode + "/batch/loss"].log(loss)

        # region LOGGING LOSS
        # Print loss progress
        print(
            f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3f} Loss "
            f"equiv: "
            f" {loss_equiv:.3} Hitrate: {hitrate:.3f}")
        # endregion

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    # Get each loss component to be logged by Neptune
    mu_loss /= total_batches
    mu_hit_rate /= total_batches
    mu_equiv_loss /= total_batches

    if mode == 'val':
        errors.append(mu_loss)
        errors_hitrate.append(mu_hit_rate)
        equiv_errors.append(mu_equiv_loss)
        np.save(f'{model_path}/errors_val.npy', errors)
        np.save(f'{model_path}/errors_hitrate.npy', errors_hitrate)
        np.save(f'{model_path}/equiv_val.npy', equiv_errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)

    # Write to standard output. Allows printing to file progress when using HPC
    sys.stdout.flush()

    if run is not None:
        run[mode + "/epoch/loss"].log(mu_loss)
        run[mode + "/epoch/hitrate"].log(mu_hit_rate)
        run[mode + "/epoch/equivariance"].log(mu_equiv_loss)



if __name__ == "__main__":
    for epoch_ in range(1, args.epochs + 1):
        train(epoch_, train_loader, 'train')
        with torch.no_grad():
            train(epoch_, val_loader, 'val')
    if run is not None:
        run.stop()
