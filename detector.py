import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TorchAEAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    PyTorch autoencoder-based anomaly detector with a scikit-learn-style interface:

    - fit(X, y):
        * Train an AE on X for reconstruction
        * If y contains both classes, use train reconstruction errors + labels to search
          the threshold maximizing train accuracy (0=normal, 1=anomaly)
        * If y is missing or single-class, fall back to a fixed quantile threshold

    - predict(X):
        * error > threshold => 1 (anomaly/malicious), otherwise 0 (normal/benign)

    - predict_proba(X):
        * Min-max normalize errors to [0,1] as a heuristic anomaly probability
          proba[:, 1] = P(anomaly)
          proba[:, 0] = P(normal)
    """

    class _AE(nn.Module):
        def __init__(self, input_dim: int, hidden_dims):
            super().__init__()
            # encoder
            enc_layers = []
            prev = input_dim
            for h in hidden_dims:
                enc_layers.append(nn.Linear(prev, h))
                enc_layers.append(nn.ReLU())
                prev = h
            self.encoder = nn.Sequential(*enc_layers)

            # decoder (symmetric)
            dec_layers = []
            hidden_dims = list(hidden_dims)
            if len(hidden_dims) > 1:
                for h in reversed(hidden_dims[:-1]):
                    dec_layers.append(nn.Linear(prev, h))
                    dec_layers.append(nn.ReLU())
                    prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out

    def __init__(
        self,
        hidden_dims=(128, 32),
        lr=1e-3,
        batch_size=256,
        epochs=20,
        weight_decay=1e-5,
        device: str = None,
        verbose: bool = False,
        unsupervised_quantile: float = 0.95,  # quantile used when labels are absent
    ):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device = device  # None -> auto-select
        self.verbose = verbose
        self.unsupervised_quantile = unsupervised_quantile

    # ---------- Core training logic ----------

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)

        # 1) Select device
        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device

        # 2) If labels exist, train AE on y==0 samples; otherwise use all
        if y is not None:
            y = np.asarray(y).astype(int)
            normal_mask = (y == 0)
            if normal_mask.any():
                X_train_ae = X[normal_mask]
            else:
                # No normal samples in training; fall back to using all samples
                X_train_ae = X
        else:
            X_train_ae = X

        # 3) Standardization: fit the scaler on the AE training set only
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train_ae)

        n_train, input_dim = X_train_scaled.shape

        # 4) Build the AE network
        self.ae_ = self._AE(input_dim, self.hidden_dims).to(self.device_)

        # 5) DataLoader (normal samples only)
        train_dataset = TensorDataset(torch.from_numpy(X_train_scaled))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 6) Train the AE
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.ae_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.ae_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0

            if self.verbose:
                iter_loader = tqdm(
                    train_loader,
                    desc=f"[AE] epoch {epoch+1}/{self.epochs}",
                    leave=False,
                )
            else:
                iter_loader = train_loader

            for (xb,) in iter_loader:
                xb = xb.to(self.device_)
                optimizer.zero_grad()
                recon = self.ae_(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= n_train
            if self.verbose:
                print(f"[AE] epoch {epoch+1}/{self.epochs}, loss={epoch_loss:.6f}")

        # 7) Reconstruction errors on the whole training set X (incl. anomalies)
        self.ae_.eval()
        with torch.no_grad():
            X_all_scaled = self.scaler_.transform(X)
            X_tensor = torch.from_numpy(X_all_scaled).to(self.device_)
            recon_all = self.ae_(X_tensor).cpu().numpy()
        errors_all = np.mean((X_all_scaled - recon_all) ** 2, axis=1)
        self.errors_train_ = errors_all

        # 8) Threshold: prefer the quantile of normal-sample errors
        if y is not None and (y == 0).any():
            normal_errors = errors_all[y == 0]
        else:
            normal_errors = errors_all

        self.error_mean_ = float(normal_errors.mean())
        self.error_std_ = float(normal_errors.std())

        # e.g., 99% quantile: about 1% of normal samples are treated as anomalies
        q = 0.99
        q = max(min(q, 0.9999), 0.9)  # clamp to a reasonable range
        self.threshold_ = float(np.quantile(normal_errors, q))

        # 9) Train-set metrics (log only)
        if y is not None:
            y_pred_train = (errors_all > self.threshold_).astype(int)
            tp = int(((y == 1) & (y_pred_train == 1)).sum())
            tn = int(((y == 0) & (y_pred_train == 0)).sum())
            fp = int(((y == 0) & (y_pred_train == 1)).sum())
            fn = int(((y == 1) & (y_pred_train == 0)).sum())
            total = len(y)

            acc = (tp + tn) / total if total > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            self.train_acc_ = float(acc)
            self.train_tpr_ = float(tpr)
            self.train_fpr_ = float(fpr)
        else:
            self.train_acc_ = None
            self.train_tpr_ = None
            self.train_fpr_ = None

        if self.verbose:
            if self.train_acc_ is not None:
                print(
                    f"[AE] device={self.device_}, "
                    f"threshold={self.threshold_:.6f}, "
                    f"train_acc={self.train_acc_:.4f}, "
                    f"train_TPR={self.train_tpr_:.4f}, "
                    f"train_FPR={self.train_fpr_:.4f}"
                )
            else:
                print(
                    f"[AE] device={self.device_}, "
                    f"threshold={self.threshold_:.6f}, "
                    f"normal_error_mean={self.error_mean_:.6f}, "
                    f"normal_error_std={self.error_std_:.6f}"
                )

        return self

    def _set_threshold_unsupervised(self, errors):
        """Unsupervised threshold: a fixed quantile."""
        self.error_mean_ = float(errors.mean())
        self.error_std_ = float(errors.std())
        q = float(self.unsupervised_quantile)
        q = max(min(q, 0.9999), 0.5)
        self.threshold_ = float(np.quantile(errors, q))
        self.train_acc_ = None
        self.train_tpr_ = None
        self.train_fpr_ = None

    def _learn_threshold_supervised(self, errors, y):
        """
        With labels, scan all thresholds via ROC to maximize train accuracy.
        Convention: y==1 is anomaly/malicious, y==0 is normal/benign.
        """
        self.error_mean_ = float(errors.mean())
        self.error_std_ = float(errors.std())

        fpr, tpr, thresholds = roc_curve(y, errors)  # larger error => more anomalous

        # roc_curve thresholds[0] is usually inf (predict-everything-normal)
        # that point may maximize accuracy but is meaningless for detection; drop it
        valid = ~np.isinf(thresholds)
        fpr = fpr[valid]
        tpr = tpr[valid]
        thresholds = thresholds[valid]

        if len(thresholds) == 0:
            # Should not happen; fall back to the unsupervised quantile strategy
            self._set_threshold_unsupervised(errors)
            return

        p_pos = float((y == 1).mean())

        # accuracy = TPR * P(pos) + (1 - FPR) * P(neg)
        acc = tpr * p_pos + (1.0 - fpr) * (1.0 - p_pos)
        best_idx = int(np.argmax(acc))

        self.threshold_ = float(thresholds[best_idx])
        self.train_acc_ = float(acc[best_idx])
        self.train_tpr_ = float(tpr[best_idx])
        self.train_fpr_ = float(fpr[best_idx])

        if self.verbose:
            print(f"[AE] supervised threshold={self.threshold_:.6f}, "
                  f"train_acc={self.train_acc_:.4f}, "
                  f"train_TPR={self.train_tpr_:.4f}, "
                  f"train_FPR={self.train_fpr_:.4f}")

    # ---------- Inference interface ----------

    def decision_function(self, X):
        """Return per-sample reconstruction error (larger = more anomalous)"""
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)

        self.ae_.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled).to(self.device_)
            recon = self.ae_(X_tensor).cpu().numpy()
        errors = np.mean((X_scaled - recon) ** 2, axis=1)
        return errors

    def predict(self, X):
        """
        Output 0/1 labels given the threshold:
            0 = normal / benign
            1 = anomaly / malicious
        """
        errors = self.decision_function(X)
        return (errors > self.threshold_).astype(int)

    def predict_proba(self, X):
        """
        Min-max normalize errors as a heuristic anomaly probability:
            proba[:, 1] = P(anomaly)
            proba[:, 0] = P(normal)
        """
        errors = self.decision_function(X)
        e_min = float(errors.min())
        e_max = float(errors.max())
        if e_max > e_min:
            e_norm = (errors - e_min) / (e_max - e_min)
        else:
            e_norm = np.zeros_like(errors)

        proba_pos = e_norm
        proba_neg = 1.0 - proba_pos
        return np.vstack([proba_neg, proba_pos]).T


class TorchN3ICDetector(BaseEstimator, ClassifierMixin):
    """
    PyTorch implementation of N3IC (sklearn-style interface):
      1) Bit-level binarization of the 20-D features following the original N3IC rules
      2) Map inputs from {0,1} to {-1,1}
      3) BNN: QuantDense -> BN -> QuantDense -> BN -> QuantDense(2)
      4) Train with squared hinge loss + Adam
    """

    N3IC_FEATURES = [
        "dur",
        "proto",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sload",
        "dload",
        "spkts",
        "dpkts",
        "smean",
        "dmean",
        "sinpkt",
        "dinpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "ct_src_ltm",
        "ct_dst_ltm",
        "ct_dst_src_ltm",
    ]

    SIZE_IN_BITS = {
        "dur": 8,
        "proto": 8,
        "sbytes": 16,
        "dbytes": 16,
        "sttl": 8,
        "dttl": 8,
        "sload": 24,
        "dload": 24,
        "spkts": 16,
        "dpkts": 16,
        "smean": 16,
        "dmean": 16,
        "sinpkt": 16,
        "dinpkt": 16,
        "tcprtt": 8,
        "synack": 8,
        "ackdat": 8,
        "ct_src_ltm": 8,
        "ct_dst_ltm": 8,
        "ct_dst_src_ltm": 8,
    }

    # indices of [sbytes, dbytes, sload, dload] within selected_columns
    SCALE_KB_INDEX = {2, 3, 6, 7}

    # alias mapping across UNSW column-name variants (key = notebook name)
    FEATURE_ALIASES = {
        "smean": ["smeansz"],
        "dmean": ["dmeansz"],
        "sinpkt": ["sintpkt"],
        "dinpkt": ["dintpkt"],
        "ct_src_ltm": ["ct_src_ ltm", "ctsrcltm"],
    }

    GENERIC_DATASETS = {"ids2017", "kdd", "malicioustls"}

    class _SignSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            out = x.sign()
            out[out == 0] = 1.0
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            grad = grad_output.clone()
            grad[x.abs() > 1] = 0
            return grad

    @classmethod
    def _ste_sign(cls, x):
        return cls._SignSTE.apply(x)

    class _QuantDense(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.uniform_(self.weight, -0.1, 0.1)

        def forward(self, x):
            x_q = TorchN3ICDetector._ste_sign(x)
            w_q = TorchN3ICDetector._ste_sign(self.weight)
            return F.linear(x_q, w_q, bias=None)

    class _BNN(nn.Module):
        def __init__(self, input_dim: int, neurons):
            super().__init__()
            n1, n2, n3 = neurons

            self.fc1 = TorchN3ICDetector._QuantDense(input_dim, n1)
            # Keras momentum=0.9 corresponds to PyTorch momentum=0.1
            self.bn1 = nn.BatchNorm1d(n1, momentum=0.1, affine=True)
            self.bn1.weight.requires_grad = False
            nn.init.ones_(self.bn1.weight)
            nn.init.zeros_(self.bn1.bias)

            self.fc2 = TorchN3ICDetector._QuantDense(n1, n2)
            self.bn2 = nn.BatchNorm1d(n2, momentum=0.1, affine=True)
            self.bn2.weight.requires_grad = False
            nn.init.ones_(self.bn2.weight)
            nn.init.zeros_(self.bn2.bias)

            self.fc3 = TorchN3ICDetector._QuantDense(n2, n3)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.fc3(x)
            # align with the notebook's final softmax
            x = F.softmax(x, dim=1)
            return x

    def __init__(
        self,
        neurons=(64, 32, 2),
        lr=1e-4,
        batch_size=256,
        epochs=15,
        val_size=0.2,
        random_state=0,
        device: str = None,
        verbose: bool = False,
    ):
        self.neurons = neurons
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_size = val_size
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.feature_names_ = None
        self.dataset_name_ = None

    def set_feature_names(self, feature_names):
        if feature_names is None:
            self.feature_names_ = None
        else:
            self.feature_names_ = list(feature_names)
        return self

    def set_dataset_name(self, dataset_name):
        self.dataset_name_ = dataset_name
        return self

    @staticmethod
    def _normalize_name(name):
        s = str(name).strip().lower()
        # ignore case, spaces, underscores and similar minor differences
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    def _normalize_dataset_name(self):
        if self.dataset_name_ is None:
            return None
        return self._normalize_name(self.dataset_name_)

    def _resolve_feature_indices(self, input_dim: int):
        if self.feature_names_ is None:
            if input_dim == len(self.N3IC_FEATURES):
                return list(range(input_dim))
            raise ValueError(
                "n3ic requires feature_names to match the fixed 20 input features; "
                "they are missing and the input dimension is not 20."
            )

        norm_names = [self._normalize_name(n) for n in self.feature_names_]
        name_to_idx = {n: i for i, n in enumerate(norm_names)}
        missing = []
        resolved_idx = []
        for base_name in self.N3IC_FEATURES:
            candidates = [base_name] + self.FEATURE_ALIASES.get(base_name, [])
            candidate_norms = [self._normalize_name(c) for c in candidates]

            hit = None
            for cn in candidate_norms:
                if cn in name_to_idx:
                    hit = name_to_idx[cn]
                    break

            if hit is None:
                missing.append(base_name)
            else:
                resolved_idx.append(hit)

        if missing:
            raise ValueError(
                f"n3ic is missing required feature columns: {missing}. "
                f"current column count={len(self.feature_names_)}"
            )

        return resolved_idx

    @staticmethod
    def _default_bit_width_for_feature(name: str):
        n = str(name).lower()
        if any(k in n for k in ["ttl", "flag", "proto", "state"]):
            return 8
        if any(k in n for k in ["duration", "dur", "iat", "pkt", "packet"]):
            return 16
        if any(k in n for k in ["byte", "load", "rate", "window"]):
            return 24
        return 16

    @staticmethod
    def _should_scale_kb(name: str):
        n = str(name).lower()
        return ("byte" in n) or ("load" in n)

    def _build_feature_profile(self, input_dim: int):
        ds = self._normalize_dataset_name()

        # UNSW-NB15: strictly follow the paper notebook's features and bit widths
        if ds in {"unswnb15"}:
            feature_idx = self._resolve_feature_indices(input_dim)
            bit_widths = [self.SIZE_IN_BITS[name] for name in self.N3IC_FEATURES]
            scale_kb_index = set(self.SCALE_KB_INDEX)
            if self.feature_names_ is not None:
                selected_names = [self.feature_names_[i] for i in feature_idx]
            else:
                selected_names = list(self.N3IC_FEATURES)
            return feature_idx, bit_widths, scale_kb_index, selected_names

        # Other datasets: auto-select the first 20 features and binarize N3IC-style
        if ds in self.GENERIC_DATASETS:
            if input_dim < 20:
                raise ValueError(f"n3ic needs at least 20 features, got {input_dim}")
            feature_idx = list(range(20))
            if self.feature_names_ is not None and len(self.feature_names_) >= 20:
                selected_names = [self.feature_names_[i] for i in feature_idx]
            else:
                selected_names = [f"f{i}" for i in feature_idx]

            bit_widths = [self._default_bit_width_for_feature(n) for n in selected_names]
            scale_kb_index = {i for i, n in enumerate(selected_names) if self._should_scale_kb(n)}
            return feature_idx, bit_widths, scale_kb_index, selected_names

        raise ValueError(
            "n3ic currently supports: unsw-nb15, ids2017, kdd, Malicious_TLS; "
            f"current dataset_name={self.dataset_name_!r}"
        )

    def _to_n3ic_binary(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_sel = X[:, self.feature_idx_]
        X_int = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64, copy=False)

        n = X_int.shape[0]
        X_bin = np.empty((n, self.total_bits_), dtype=np.float32)

        ptr = 0
        for j, bits in enumerate(self.bit_widths_):
            col = X_int[:, j].copy()
            if j in self.scale_kb_index_:
                col = col // 1000

            max_v = (1 << bits) - 1
            col = np.clip(col, 0, max_v)

            shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
            block = ((col[:, None] >> shifts[None, :]) & 1).astype(np.float32)
            X_bin[:, ptr:ptr + bits] = block
            ptr += bits

        X_bin[X_bin == 0.0] = -1.0
        return X_bin

    @staticmethod
    def _squared_hinge_loss(y_true_onehot, y_pred):
        y_true_sign = 2.0 * y_true_onehot - 1.0
        margin = 1.0 - y_true_sign * y_pred
        return torch.mean(torch.square(torch.clamp(margin, min=0.0)))

    def _clip_quant_weights(self):
        for module in self.model_.modules():
            if isinstance(module, self._QuantDense):
                module.weight.data.clamp_(-1.0, 1.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"n3ic supports binary classification only, got {classes.tolist()}")
        self.classes_ = classes

        y_enc = np.searchsorted(self.classes_, y)

        self.feature_idx_, self.bit_widths_, self.scale_kb_index_, self.selected_feature_names_ = (
            self._build_feature_profile(X.shape[1])
        )
        self.total_bits_ = int(sum(self.bit_widths_))

        if self.verbose:
            print(f"[N3IC] dataset={self.dataset_name_}, selected_features={self.selected_feature_names_}")

        X_bin = self._to_n3ic_binary(X)

        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device

        idx = np.arange(len(y_enc))
        use_val = self.val_size is not None and self.val_size > 0 and len(y_enc) >= 10
        if use_val:
            try:
                train_idx, val_idx = train_test_split(
                    idx,
                    test_size=self.val_size,
                    random_state=self.random_state,
                    stratify=y_enc,
                )
            except ValueError:
                # stratify may fail under extreme class imbalance; fall back to plain split
                train_idx, val_idx = train_test_split(
                    idx,
                    test_size=self.val_size,
                    random_state=self.random_state,
                    stratify=None,
                )
        else:
            train_idx, val_idx = idx, None

        if len(train_idx) < 2:
            raise ValueError("n3ic needs at least 2 training samples.")

        X_train = torch.from_numpy(X_bin[train_idx]).float()
        y_train = torch.from_numpy(y_enc[train_idx]).long()
        drop_last = (len(train_idx) % self.batch_size == 1)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        self.model_ = self._BNN(input_dim=self.total_bits_, neurons=self.neurons).to(self.device_)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        if val_idx is not None:
            X_val = torch.from_numpy(X_bin[val_idx]).float().to(self.device_)
            y_val = torch.from_numpy(y_enc[val_idx]).long().to(self.device_)
        else:
            X_val, y_val = None, None

        best_state = None
        best_val_acc = -np.inf

        for epoch in range(self.epochs):
            self.model_.train()
            running_loss = 0.0
            total_train = 0

            for xb, yb in train_loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad()
                pred = self.model_(xb)
                y_onehot = F.one_hot(yb, num_classes=2).float()
                loss = self._squared_hinge_loss(y_onehot, pred)
                loss.backward()
                optimizer.step()
                self._clip_quant_weights()

                running_loss += loss.item() * xb.size(0)
                total_train += xb.size(0)

            train_loss = running_loss / max(total_train, 1)

            self.model_.eval()
            with torch.no_grad():
                if X_val is not None:
                    val_pred = self.model_(X_val).argmax(dim=1)
                    val_acc = (val_pred == y_val).float().mean().item()
                else:
                    X_tr = X_train.to(self.device_)
                    y_tr = y_train.to(self.device_)
                    tr_pred = self.model_(X_tr).argmax(dim=1)
                    val_acc = (tr_pred == y_tr).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}

            if self.verbose:
                print(
                    f"[N3IC] epoch {epoch + 1}/{self.epochs}, "
                    f"loss={train_loss:.6f}, best_acc={best_val_acc:.4f}"
                )

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.train_best_acc_ = float(best_val_acc)
        return self

    def predict_proba(self, X):
        X_bin = self._to_n3ic_binary(X)
        X_tensor = torch.from_numpy(X_bin).float().to(self.device_)

        self.model_.eval()
        with torch.no_grad():
            proba = self.model_(X_tensor).cpu().numpy()
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.classes_[pred_idx]
