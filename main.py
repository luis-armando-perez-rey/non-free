import torch.utils.data
from torch import save
import pickle

from models.models_nn import *
from utils.nn_utils import *
from utils.parse_args import get_args
from models.resnet import *

# Import datasets
from datasets.equiv_dset import *

torch.cuda.empty_cache()

parser = get_args()
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)



# Save paths
MODEL_PATH = os.path.join(args.checkpoints_dir, args.model_name)

figures_dir = os.path.join(MODEL_PATH, 'figures')
model_file = os.path.join(MODEL_PATH, 'model.pt')
meta_file = os.path.join(MODEL_PATH, 'metadata.pkl')
log_file = os.path.join(MODEL_PATH, 'log.txt')

make_dir(MODEL_PATH)
make_dir(figures_dir)

pickle.dump({'args': args}, open(meta_file, 'wb'))

# Set dataset
if args.dataset == 'rot-square':
    dset = EquivDataset(f'{args.data_dir}/square/')
elif args.dataset == 'rot-arrows':
    dset = EquivDataset(f'{args.data_dir}/arrows/')
else:
    raise ValueError(f'Dataset {args.dataset} not supported')


dset, dset_test = torch.utils.data.random_split(dset, [len(dset) - int(len(dset)/10), int(len(dset)/10)])
train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dset_test,
                                            batch_size=args.batch_size, shuffle=True)


print("# train set:", len(dset))
print("# test set:", len(dset_test))

# Sample data
img, _, acn = next(iter(train_loader))
img_shape = img.shape[1:]

N = args.num
extra_dim = args.extra_dim

model = MDN(img_shape[0], 2, N, extra_dim, model=args.model).to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
errors = []

def train(epoch, data_loader, mode='train'):
    mu_loss = 0
    global_step = len(data_loader) * epoch

    for batch_idx, (img, img_next, action) in enumerate(data_loader):
        batch_size = img.shape[0]
        global_step += batch_size

        if mode == 'train':
            optimizer.zero_grad()
            model.train()

        elif mode == 'val':
            model.eval()


        img = img.to(device)
        img_next = img_next.to(device)
        action = action.to(device).squeeze(1)

        z_mean, z_logvar, extra = model(img)
        z_mean_next, z_logvar_next, extra_next = model(img_next)
        z_logvar = -4.6*torch.ones(z_logvar.shape).to(z_logvar.device)
        z_logvar_next = -4.6*torch.ones(z_logvar.shape).to(z_logvar.device)

        s = torch.sin(action)
        c = torch.cos(action)
        rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).permute((2,0,1)).unsqueeze(1)

        z_mean_pred = (rot @ z_mean.unsqueeze(-1).detach()).squeeze(-1)  #Beware the detach!!!

        #Probabilistic losses (cross-entropy, Chamfer etc)
        #loss_equiv = prob_loss(z_mean_next, z_logvar_next, z_mean_pred, z_logvar, N)
        #loss_equiv = prob_loss(z_mean_pred, z_logvar, z_mean_next, z_logvar_next, N)
        loss_equiv = ((z_mean_pred.unsqueeze(1) - z_mean_next.unsqueeze(2))**2).sum(-1).min(dim=-1)[0].sum(dim=-1).mean() #Chamfer/Hausdorff loss

        if extra_dim > 0:
            #Contrastive Loss: infoNCE w/ cosine similariy
            distance_matrix = (extra.unsqueeze(1) * extra_next.unsqueeze(0)).sum(-1) / args.tau
            loss_contra = -torch.mean((extra * extra_next).sum(-1) / args.tau - torch.logsumexp(distance_matrix, dim=-1))
            loss = loss_equiv + loss_contra
        else:
            loss = loss_equiv


        print(f"{mode.upper()} Epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3}")

        mu_loss += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

    mu_loss /= len(data_loader)



    if mode == 'val':
        errors.append(mu_loss)
        np.save(f'{MODEL_PATH}/errors_val.npy', errors)

        if (epoch % args.save_interval) == 0:
            save(model, model_file)




if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, 'train')
        with torch.no_grad():
            train(epoch, val_loader, 'val')
