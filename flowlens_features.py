#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flowlens_features.py

Extract FlowLens-style features from PCAP:
  PCAP directories (classes by sub-directory) ->
  flows (5-tuple) ->
  packet-length histogram ->
  X_all, y_all, meta_all

Dependencies:
    pip install dpkt numpy
"""
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import dpkt
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

# ===================== Basic types =====================

# FlowKey: (ip1, port1, ip2, port2, proto)
FlowKey = Tuple[str, int, str, int, int]


@dataclass
class FlowMarkerConfig:
    """
    FlowLens-style configuration: quantization parameters etc.
    """
    ql_pl: int = 4          # packet-length quantization: bin_size = 2^QL
    max_pl: int = 1500      # maximum packet length considered (bytes)
    use_ipt: bool = False   # whether to use inter-arrival time
    ql_ipt: int = 6         # IPT quantization (only when use_ipt=True)

    @property
    def pl_bin_size(self) -> int:
        return 1 << self.ql_pl

    @property
    def num_pl_bins(self) -> int:
        # ceil(max_pl / bin_size)
        return (self.max_pl + self.pl_bin_size - 1) // self.pl_bin_size


# ===================== Utility functions =====================

def inet_to_str(x: bytes) -> str:
    """IPv4/IPv6 bytes -> string."""
    try:
        return socket.inet_ntop(socket.AF_INET, x)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, x)


def normalize_5tuple(src_ip: str, sport: int,
                     dst_ip: str, dport: int,
                     proto: int) -> FlowKey:
    """
    Normalize a bidirectional flow to the same key: sort both ends by (ip, port).
    This way packets of both directions of a TCP connection share one flow.
    """
    a = (src_ip, sport)
    b = (dst_ip, dport)
    if a <= b:
        ip1, p1, ip2, p2 = src_ip, sport, dst_ip, dport
    else:
        ip1, p1, ip2, p2 = dst_ip, dport, src_ip, sport
    return ip1, p1, ip2, p2, proto


def _process_one_pcap(args):
    """
    Helper for multiprocessing:
      input: (pcap_path_str, label_id, class_name, pcap_root_str, cfg)
      output: (X_part: List[np.ndarray], y_part: List[int], meta_part: List[dict])
    """
    pcap_path_str, label_id, class_name, pcap_root_str, cfg = args
    pcap_path = Path(pcap_path_str)

    # parse all flows in this pcap
    flows = read_pcap_flows(str(pcap_path))
    if not flows:
        return [], [], []

    # FlowLens features
    markers = build_flow_markers(flows, cfg)

    X_part: List[np.ndarray] = []
    y_part: List[int] = []
    meta_part: List[dict] = []

    for flow_key, feat in markers.items():
        X_part.append(feat.astype(np.float32))
        y_part.append(label_id)
        meta_part.append({
            "flow_key": flow_key,            # (ip1,p1,ip2,p2,proto)
            "pcap_path": str(pcap_path),     # for traceability
            "class_name": class_name,        # original directory name (fine-grained type)
            "binary_label": label_id,        # 0/1 binary label
        })

    return X_part, y_part, meta_part


# ===================== PCAP -> flows =====================

def read_pcap_flows(pcap_path: str) -> Dict[FlowKey, List[Tuple[float, int]]]:
    """
    Parse flows from a single PCAP:
        flow_key -> [(timestamp, packet_length), ...]
    packet_length uses the IP-layer length.
    """
    flows: Dict[FlowKey, List[Tuple[float, int]]] = {}

    with open(pcap_path, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue

            ip = eth.data
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue

            l4 = ip.data
            if not isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue

            src_ip = inet_to_str(ip.src)
            dst_ip = inet_to_str(ip.dst)
            sport = int(l4.sport)
            dport = int(l4.dport)
            proto = int(ip.p)

            key = normalize_5tuple(src_ip, sport, dst_ip, dport, proto)

            try:
                pkt_len = int(ip.len)
            except AttributeError:
                pkt_len = len(buf)

            if key not in flows:
                flows[key] = []
            flows[key].append((float(ts), pkt_len))

    # sort by time
    for k in list(flows.keys()):
        pkt_list = flows[k]
        if not pkt_list:
            del flows[k]
        else:
            pkt_list.sort(key=lambda x: x[0])

    return flows


# ===================== flows -> FlowLens marker =====================

def build_flow_markers(
    flows: Dict[FlowKey, List[Tuple[float, int]]],
    cfg: FlowMarkerConfig,
) -> Dict[FlowKey, np.ndarray]:
    """
    (ts, len) sequence of a flow -> FlowLens-style marker:
      - packet-length histogram
      - if use_ipt=True, append the inter-arrival time histogram
    """
    num_pl_bins = cfg.num_pl_bins
    markers: Dict[FlowKey, np.ndarray] = {}

    for key, pkt_list in flows.items():
        # 1) packet-length histogram
        pl_hist = np.zeros(num_pl_bins, dtype=np.uint16)
        for ts, length in pkt_list:
            l = min(length, cfg.max_pl)
            bin_idx = l >> cfg.ql_pl  # equivalent to l // (2**ql_pl)
            if bin_idx >= num_pl_bins:
                bin_idx = num_pl_bins - 1
            pl_hist[bin_idx] += 1

        if not cfg.use_ipt:
            markers[key] = pl_hist.astype(np.float32)
            continue

        # 2) IPT histogram if requested; a simple implementation
        times = np.array([ts for ts, _ in pkt_list], dtype=np.float64)
        if len(times) > 1:
            deltas = np.diff(times)          # seconds
            deltas_ms = deltas * 1000.0      # milliseconds
            max_ms = 3600 * 1000.0           # clamp to 1 hour
            bin_size = 1 << cfg.ql_ipt
            num_ipt_bins = int((max_ms + bin_size - 1) // bin_size)

            ipt_hist = np.zeros(num_ipt_bins, dtype=np.uint16)
            for dt in deltas_ms:
                if dt < 0:
                    continue
                if dt > max_ms:
                    idx = num_ipt_bins - 1
                else:
                    idx = int(dt) >> cfg.ql_ipt
                    if idx >= num_ipt_bins:
                        idx = num_ipt_bins - 1
                ipt_hist[idx] += 1
        else:
            ipt_hist = np.zeros(1, dtype=np.uint16)

        marker = np.concatenate([pl_hist, ipt_hist]).astype(np.float32)
        markers[key] = marker

    return markers


# ===================== Main interface: load from directories (plan B) =====================

def load_flowlens_dataset(
    dataset_name: str,
    num_workers: int = 4,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Following the style of utils.load_full_dataset:
        input: dataset_name
        output: X_all, y_all, meta_all

    Plan B: the first-level sub-directory name under the PCAP root is the raw class name,
    and the final label is binary:
        0 = benign
        1 = malicious

    Enhanced version:
      - Extract features from pcaps once and cache under .cache_flowlens
      - Later runs load the cache directly, skipping re-extraction
      - Multiprocessing + tqdm progress bar during extraction
    """

    # 1) PCAP root paths per dataset (adjust to your environment)
    PCAP_ROOT_MAP = {
        "unsw-nb15": Path("/data1/lx/dataset/unsw-nb15/PCAP"),
        "ids2017":   Path("/data1/lx/dataset/CICIDS2017/"),
        "dapt-2020": Path("/data1/lx/dataset/dapt-2020/"),
        # add more datasets if needed
    }

    if dataset_name not in PCAP_ROOT_MAP:
        raise ValueError(f"[FlowLens] dataset {dataset_name} not configured in PCAP_ROOT_MAP")

    pcap_root = PCAP_ROOT_MAP[dataset_name]
    if not pcap_root.exists():
        raise FileNotFoundError(f"[FlowLens] PCAP root does not exist: {pcap_root}")

    # 2) cache directory and file
    if cache_dir is None:
        cache_dir = pcap_root / ".cache_flowlens"
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_cache_path = cache_dir / f"{dataset_name}_X_all.npy"
    y_cache_path = cache_dir / f"{dataset_name}_y_all.npy"
    meta_cache_path = cache_dir / f"{dataset_name}_meta_all.pkl"

    # === 2.1 load the cache if it exists ===
    if X_cache_path.exists() and y_cache_path.exists() and meta_cache_path.exists():
        print(f"[FlowLens] Found cached features in {cache_dir}, loading ...")
        X_all = np.load(X_cache_path)
        y_all = np.load(y_cache_path)
        with open(meta_cache_path, "rb") as f:
            meta_all = pickle.load(f)
        print(f"[FlowLens] Loaded from cache: X_all.shape = {X_all.shape}, "
              f"label dist = {Counter(y_all)}")
        return X_all, y_all, meta_all

    # 3) directory names treated as benign; everything else is malicious
    BENIGN_DIRS_MAP = {
        "unsw-nb15": {"benign", "normal", "Benign", "BENIGN"},
        "ids2017":   {"BENIGN", "Benign", "benign", "normal"},
        "dapt-2020": {"benign", "normal", "Benign", "BENIGN"},
    }
    benign_dirs = BENIGN_DIRS_MAP.get(
        dataset_name,
        {"benign", "normal", "Benign", "BENIGN"}
    )

    # 4) walk all pcaps recursively
    pcap_paths = sorted(pcap_root.rglob("*.pcap"))
    if not pcap_paths:
        raise RuntimeError(f"[FlowLens] no .pcap files found under {pcap_root}")

    cfg = FlowMarkerConfig(ql_pl=4, use_ipt=False)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_all: List[dict] = []

    benign_class_names = set()
    malicious_class_names = set()

    print(f"[FlowLens] Start featurizing {len(pcap_paths)} pcap files ...")

    # 4.1 decide each pcap's label / class_name from its directory in the main process
    jobs = []
    for pcap_path in pcap_paths:
        rel = pcap_path.relative_to(pcap_root)
        if len(rel.parts) < 2:
            # pcaps directly under the root lack class info; skip or default them
            print(f"[FlowLens] Skip {pcap_path} (no class subdir under root)")
            continue

        class_name = rel.parts[0]  # first-level sub-directory (raw class)

        # map to binary labels: 0=benign, 1=malicious
        if class_name in benign_dirs:
            label_id = 0
            benign_class_names.add(class_name)
        else:
            label_id = 1
            malicious_class_names.add(class_name)

        jobs.append((str(pcap_path), label_id, class_name, str(pcap_root), cfg))

    if not jobs:
        raise RuntimeError("[FlowLens] no valid pcap tasks; check the directory layout.")

    # 4.2 process all pcaps with multiprocessing
    if num_workers is None or num_workers <= 0:
        num_workers = os.cpu_count() or 4

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_process_one_pcap, job) for job in jobs]

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="[FlowLens] PCAPs"):
            X_part, y_part, meta_part = fut.result()
            if not X_part:
                continue
            X_list.extend(X_part)
            y_list.extend(y_part)
            meta_all.extend(meta_part)

    if not X_list:
        raise RuntimeError("[FlowLens] no features extracted from any pcap; check the directory layout or pcap content.")

    X_all = np.stack(X_list, axis=0)
    y_all = np.array(y_list, dtype=np.int64)

    print(f"[FlowLens] Benign dirs   (→ 0): {sorted(benign_class_names)}")
    print(f"[FlowLens] Malicious dirs(→ 1): {sorted(malicious_class_names)}")
    print(f"[FlowLens] X_all.shape = {X_all.shape}, label dist = {Counter(y_all)}")

    # === 5. write the cache ===
    print(f"[FlowLens] Saving features to cache dir {cache_dir} ...")
    np.save(X_cache_path, X_all)
    np.save(y_cache_path, y_all)
    with open(meta_cache_path, "wb") as f:
        pickle.dump(meta_all, f)
    print("[FlowLens] Cache saved.")

    return X_all, y_all, meta_all