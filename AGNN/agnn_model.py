"""
agnn_model.py

Model code for the AGNN subquestion scripts (01 to 04) 

Part 1 is a copy of the model classes from the original implementation by
Phung et al. (2022), "Unsupervised air quality interpolation with attentive
graph neural network" (repo: Unsupervised-Air-Quality-Estimation). The layer
and model logic is unchanged. 

Part 2 contains the training helpers, copied from the repo's
src/modules/train/train.py and test.py with the same edits.

Part 3 is the FoldDataset class. It replaces the repo's AQDataSet with the
leave one station out protocol of this thesis (year based split, target hour
lists, drop the farthest pool station at test time). The sample format is
identical to AQDataSet: {"X", "Y", "G", "l", "climate"}.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset



# small utilities 


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, path):
    torch.save({"model_dict": model.state_dict()}, path)


def load_model(model, path):
    # map_location so models trained on a GPU also load on a CPU machine
    model.load_state_dict(torch.load(path, map_location="cpu")["model_dict"])


class EarlyStopping:
    # same behaviour as the repo: lower score is better, patience counts
    # epochs without improvement, the best model is saved to `path`
    def __init__(self, patience=5, delta=0.0, path="checkpoint.pt", verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            save_model(model, self.path)
        elif score + self.delta > self.best_score:
            self.counter += 1
            if self.verbose:
                print("   early stopping counter: %d out of %d" % (self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            save_model(model, self.path)
            self.counter = 0



# Part 1: model classes from Phung et al. (2022)


class GCN(nn.Module):
    def __init__(self, infea, outfea, act="relu", bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(infea, outfea, bias=False)
        self.act = nn.ReLU() if act == "relu" else nn.ReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(outfea))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN_2_layers(nn.Module):
    def __init__(self, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(GCN_2_layers, self).__init__()
        self.gcn_1 = GCN(hid_ft1, hid_ft2, act)
        self.gcn_2 = GCN(hid_ft2, out_ft, act)

    def forward(self, x, adj):
        x = self.gcn_1(x, adj)
        x = self.gcn_2(x, adj)
        return x


class TemporalGCN(nn.Module):
    # T-GCN style gated recurrent cell over graph convolutions
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(TemporalGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.conv_z = GCN_2_layers(hid_ft1=in_channels, hid_ft2=hidden_dim, out_ft=out_channels)
        self.linear_z = nn.Linear(2 * out_channels, out_channels)
        self.conv_r = GCN_2_layers(hid_ft1=in_channels, hid_ft2=hidden_dim, out_ft=out_channels)
        self.linear_r = nn.Linear(2 * out_channels, out_channels)
        self.conv_h = GCN_2_layers(hid_ft1=in_channels, hid_ft2=hidden_dim, out_ft=out_channels)
        self.linear_h = nn.Linear(2 * out_channels, out_channels)

    def forward(self, X, adj, H):
        Z = torch.sigmoid(self.linear_z(torch.cat([self.conv_z(X, adj), H], axis=2)))
        R = torch.sigmoid(self.linear_r(torch.cat([self.conv_r(X, adj), H], axis=2)))
        H_tilde = torch.tanh(self.linear_h(torch.cat([self.conv_h(X, adj), H * R], axis=2)))
        H = Z * H + (1 - Z) * H_tilde
        return H


class Attention_Encoder(nn.Module):
    def __init__(self, in_ft, hid_ft1, hid_ft2, out_ft, act="relu"):
        super(Attention_Encoder, self).__init__()
        self.in_dim = hid_ft1
        self.hid_dim = hid_ft2
        self.out_dim = out_ft
        self.fc = nn.Linear(in_ft, hid_ft1)
        self.rnn_gcn = TemporalGCN(hid_ft1, out_ft, hid_ft2)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: [batch, seq, n_stations, in_ft], adj: [batch, seq, n, n]
        x = self.relu(self.fc(x))
        raw_shape = x.shape
        # edit 1: use the input's device instead of hard coded "cuda"
        h = torch.zeros(raw_shape[0], raw_shape[2], self.out_dim, device=x.device)
        list_h = []
        for i in range(raw_shape[1]):
            x_i = x[:, i, :, :]
            e = adj[:, i, :, :]
            h = self.rnn_gcn(x_i, e, h)
            list_h.append(h)
        h_ = torch.stack(list_h, dim=1)
        return h_


class Discriminator(nn.Module):
    def __init__(self, h_ft, x_ft, hid_ft):
        super(Discriminator, self).__init__()
        self.fc = nn.Bilinear(h_ft, x_ft, out_features=hid_ft)
        self.linear = nn.Linear(hid_ft, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, h, x, x_c):
        ret1 = self.linear(self.relu(self.fc(h, x)))
        ret2 = self.linear(self.relu(self.fc(h, x_c)))
        ret = torch.cat((ret1, ret2), 2)
        return self.sigmoid(ret)


class Attention_STDGI(nn.Module):
    def __init__(self, in_ft, out_ft, en_hid1, en_hid2, dis_hid,
                 stdgi_noise_min=0.4, stdgi_noise_max=0.7):
        super(Attention_STDGI, self).__init__()
        self.encoder = Attention_Encoder(
            in_ft=in_ft, hid_ft1=en_hid1, hid_ft2=en_hid2, out_ft=out_ft
        )
        self.disc = Discriminator(x_ft=in_ft, h_ft=out_ft, hid_ft=dis_hid)
        self.stdgi_noise_min = stdgi_noise_min
        self.stdgi_noise_max = stdgi_noise_max

    def forward(self, x, x_k, adj):
        h = self.encoder(x, adj)
        x_c = self.corrupt(x_k)
        ret = self.disc(h[:, -1, :, :], x_k[:, -1, :, :], x_c[:, -1, :, :])
        return ret

    def corrupt(self, X):
        nb_nodes = X.shape[1]
        idx = np.random.permutation(nb_nodes)
        shuf_fts = X[:, idx, :]
        return np.random.uniform(self.stdgi_noise_min, self.stdgi_noise_max) * shuf_fts

    def embedd(self, x, adj):
        return self.encoder(x, adj)


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, key, query, mask=None):
        n_station = key.shape[1]
        query = query.unsqueeze(1)
        score = torch.bmm(query, key.transpose(1, 2))
        attn = self.softmax(score.view(-1, n_station))
        return attn


class Local_Global_Decoder(nn.Module):
    def __init__(self, in_ft, out_ft, n_layers_rnn=1, rnn="GRU",
                 cnn_hid_dim=128, fc_hid_dim=64, n_features=7, num_input_stat=7):
        super(Local_Global_Decoder, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.n_layers_rnn = n_layers_rnn
        self.num_input_stat = num_input_stat - 1
        self.embed = nn.Linear(n_features, cnn_hid_dim)
        self.linear = nn.Linear(in_features=cnn_hid_dim * 3, out_features=fc_hid_dim)
        self.linear2 = nn.Linear(fc_hid_dim, out_ft)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_ft, cnn_hid_dim)
        self.query_local = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.value_local = nn.Linear(num_input_stat - 1, cnn_hid_dim)
        self.atten = DotProductAttention()
        self.query = nn.Linear(cnn_hid_dim * 2, cnn_hid_dim)
        self.key = nn.Linear(cnn_hid_dim, cnn_hid_dim)
        self.value = nn.Linear(cnn_hid_dim, cnn_hid_dim)

    def forward(self, x, h, l, climate):
        # x: [batch, seq, n_in, features], h: [batch, seq, n_in, latent]
        # l: [batch, n_in], climate: [batch, n_climate]
        x = x[:, -1, :, :]
        h = h[:, -1, :, :]
        l_ = l.unsqueeze(2)
        x_ = torch.cat((x, h), dim=-1)
        ret = self.relu(self.fc(x_))
        ret_ = ret.permute(0, 2, 1)
        interpolation_ = torch.bmm(ret_, l_)
        interpolation_ = interpolation_.reshape(ret.shape[0], -1)
        embed = self.embed(climate)

        query_local = self.query_local(torch.cat((interpolation_, embed), dim=-1))
        value_local = self.value_local(ret_)
        key_local = self.key_local(ret_)
        atten_weight_local = self.atten(key_local, query_local)
        atten_vector_local = torch.bmm(atten_weight_local.unsqueeze(1), value_local).squeeze(1)

        query = self.query(torch.cat((interpolation_, embed), dim=-1))
        value = self.value(ret)
        key = self.key(ret)
        atten_weight = self.atten(key, query)
        atten_vector = torch.bmm(atten_weight.unsqueeze(1), value).squeeze(1)

        ret = self.linear(torch.cat((atten_vector_local, atten_vector, embed), dim=-1))
        ret = self.relu(ret)
        ret = self.linear2(ret)
        return ret



# Part 2: training helpers, copied from the repo's train/test modules


def train_stdgi_one_epoch(stdgi, dataloader, optim_e, optim_d, bce_loss, device, n_steps=2):
    # unsupervised contrastive training of the encoder
    # train_atten_stdgi: n_steps discriminator updates, then one encoder update
    epoch_loss = 0.0
    stdgi.train()
    for data in dataloader:
        for i in range(n_steps):
            optim_d.zero_grad()
            x = data["X"].to(device).float()
            G = data["G"][:, :, :, :, 0].to(device).float()
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            d_loss = bce_loss(output, lbl)
            d_loss.backward()
            optim_d.step()

        optim_e.zero_grad()
        x = data["X"].to(device).float()
        G = data["G"][:, :, :, :, 0].to(device).float()
        output = stdgi(x, x, G)
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        e_loss = bce_loss(output, lbl)
        e_loss.backward()
        optim_e.step()
        epoch_loss += e_loss.detach().cpu().item()
    return epoch_loss / len(dataloader)


def train_decoder_one_epoch(stdgi, decoder, dataloader, mse_loss, optimizer, device):
    # supervised decoder training, the repo's train_atten_decoder_fn.
    # the encoder is frozen: its embeddings are computed under no_grad, which
    # changes nothing numerically (its parameters are not in the optimizer)
    # and saves the wasted backward pass through the encoder.
    decoder.train()
    epoch_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()
        y_grt = data["Y"].to(device).float()
        x = data["X"].to(device).float()
        G = data["G"][:, :, :, :, 0].to(device).float()
        l = data["l"].to(device).float()
        cli = data["climate"].to(device).float()
        with torch.no_grad():
            h = stdgi.embedd(x, G)
        y_prd = decoder(x, h, l, cli)
        batch_loss = mse_loss(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    return epoch_loss / len(dataloader)


def decoder_val_loss(stdgi, decoder, dataloader, mse_loss, device):
    # validation loss in scaled space, the repo's test_atten_decoder_fn with test=False
    decoder.eval()
    stdgi.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            y_grt = data["Y"].to(device).float()
            x = data["X"].to(device).float()
            G = data["G"][:, :, :, :, 0].to(device).float()
            l = data["l"].to(device).float()
            cli = data["climate"].to(device).float()
            h = stdgi.embedd(x, G)
            y_prd = decoder(x, h, l, cli)
            loss = mse_loss(torch.squeeze(y_prd), torch.squeeze(y_grt))
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def predict_decoder(stdgi, decoder, dataloader, device):
    # returns the scaled predictions as one numpy array, in dataloader order
    decoder.eval()
    stdgi.eval()
    parts = []
    with torch.no_grad():
        for data in dataloader:
            x = data["X"].to(device).float()
            G = data["G"][:, :, :, :, 0].to(device).float()
            l = data["l"].to(device).float()
            cli = data["climate"].to(device).float()
            h = stdgi.embedd(x, G)
            y_prd = decoder(x, h, l, cli)
            parts.append(y_prd.detach().cpu().numpy().reshape(-1))
    return np.concatenate(parts)


# Part 3: the fold dataset (replaces the repo's AQDataSet)


class FoldDataset(Dataset):
    """
    One LOSO fold worth of data in the sample format the model expects.

    mode="train": every item is one target hour from the training years. A
      random pool station is the target, the other pool stations are the
      inputs (this is the repo's training scheme). Y is the scaled PM2.5 of
      the target station at the last hour of the window.
    mode="val":   deterministic grid of (validation hour, target station)
      pairs. The target is a pool station, the inputs are the other pool
      stations, exactly like a training sample but without randomness.
    mode="test":  every item is one test hour. The inputs are the pool
      stations minus the one farthest from the held out target. The climate
      vector comes from the held out target itself. Y is a dummy zero, the
      scripts evaluate against the raw ref_pm25 instead.

    Arrays:
      node_scaled  [T, P, n_node]  scaled node features of the pool stations
      y_scaled     [T, P]          scaled (never clipped) PM2.5 of the pool
      clim_scaled  [T, P, n_clim]  scaled climate features of the pool
      clim_target  [T, n_clim]     scaled climate features of the held out target
      dist_pool    [P, P]          pairwise distances between pool stations, km
      dist_target  [P]             distance of every pool station to the target, km
      t_list                       target hour indices for this mode
      pair_list    (val only)      list of (t, target_pool_index) pairs
    """

    def __init__(self, mode, node_scaled, y_scaled, clim_scaled, clim_target,
                 dist_pool, dist_target, seq_len, t_list, pair_list=None,
                 dist_floor_km=0.1):
        self.mode = mode
        self.node = node_scaled
        self.y = y_scaled
        self.clim = clim_scaled
        self.clim_target = clim_target
        self.dist_pool = dist_pool
        self.dist_target = dist_target
        self.seq_len = seq_len
        self.t_list = list(t_list)
        self.pair_list = pair_list
        self.floor = dist_floor_km
        self.n_pool = node_scaled.shape[1]
        self.adj_cache = {}
        self.l_cache = {}
        if mode == "test":
            # the farthest pool station sits out so the input count matches training
            self.dropped = int(np.argmax(dist_target))
            self.test_inputs = []
            for j in range(self.n_pool):
                if j != self.dropped:
                    self.test_inputs.append(j)

    def make_adjacency(self, input_idx):
        # the repo's get_adjacency_matrix: rows are inverse distance, the self
        # distance gets +15 km so 1/d stays finite, every row normalised to sum 1
        key = tuple(input_idx)
        if key in self.adj_cache:
            return self.adj_cache[key]
        n = len(input_idx)
        A = np.zeros((n, n), dtype=np.float32)
        for a in range(n):
            d = self.dist_pool[input_idx[a], input_idx].astype(np.float64).copy()
            d[a] = d[a] + 15.0
            d = np.clip(d, self.floor, None)
            w = 1.0 / d
            A[a, :] = (w / w.sum()).astype(np.float32)
        A = np.repeat(A[None, :, :], self.seq_len, axis=0)
        self.adj_cache[key] = A
        return A

    def make_l(self, input_idx, target_kind, target_j=None):
        # the repo's get_reverse_distance_matrix: normalised inverse distance
        # from every input station to the target
        key = (tuple(input_idx), target_kind, target_j)
        if key in self.l_cache:
            return self.l_cache[key]
        if target_kind == "pool":
            d = self.dist_pool[input_idx, target_j].astype(np.float64).copy()
        else:
            d = self.dist_target[list(input_idx)].astype(np.float64).copy()
        d = np.clip(d, self.floor, None)
        w = 1.0 / d
        l = (w / w.sum()).astype(np.float32)
        self.l_cache[key] = l
        return l

    def __len__(self):
        if self.mode == "val":
            return len(self.pair_list)
        return len(self.t_list)

    def __getitem__(self, index):
        if self.mode == "train":
            t = self.t_list[index]
            target = random.randrange(self.n_pool)
            inputs = []
            for j in range(self.n_pool):
                if j != target:
                    inputs.append(j)
            x = self.node[t - self.seq_len + 1: t + 1, inputs, :]
            y = np.array([self.y[t, target]], dtype=np.float32)
            climate = self.clim[t, target, :]
            G = self.make_adjacency(inputs)
            l = self.make_l(inputs, "pool", target)
        elif self.mode == "val":
            t, target = self.pair_list[index]
            inputs = []
            for j in range(self.n_pool):
                if j != target:
                    inputs.append(j)
            x = self.node[t - self.seq_len + 1: t + 1, inputs, :]
            y = np.array([self.y[t, target]], dtype=np.float32)
            climate = self.clim[t, target, :]
            G = self.make_adjacency(inputs)
            l = self.make_l(inputs, "pool", target)
        else:
            t = self.t_list[index]
            inputs = self.test_inputs
            x = self.node[t - self.seq_len + 1: t + 1, inputs, :]
            y = np.array([0.0], dtype=np.float32)
            climate = self.clim_target[t, :]
            G = self.make_adjacency(inputs)
            l = self.make_l(inputs, "heldout")

        sample = {
            "X": np.ascontiguousarray(x),
            "Y": y,
            "l": l,
            "climate": np.ascontiguousarray(climate),
            "G": np.expand_dims(G, -1),
        }
        return sample
