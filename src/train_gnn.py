"""Train one (architecture, seed, stage) configuration of the multi-task glioma model."""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mdata, models
from feat import TASKS


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="pretrain")          # pretrain | ft_none | ft_np | ft_npw
    p.add_argument("--init", default="")                   # checkpoint to warm-start from
    p.add_argument("--ft_mode", default="none")            # none | np_subset | np_weight
    p.add_argument("--no_putative", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tag", default="")
    return p.parse_args()


def run_eval(model, loader):
    model.eval()
    P, Y, M, I = [], [], [], []
    with torch.no_grad():
        for b in loader:
            P.append(torch.sigmoid(model(b)).numpy())
            Y.append(b.y.numpy()); M.append(b.mask.numpy()); I.append(b.idx.numpy())
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(M), np.concatenate(I))


def val_loss(model, loader, pw):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            l = models.masked_bce(model(b), b.y, b.mask, pw)
            k = float(b.mask.sum())
            tot += float(l) * k; n += k
    return tot / max(n, 1)


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = a.tag or f"{a.arch}_s{a.seed}_{a.stage}"
    t0 = time.time()

    df, ds = mdata.chembl_dataset()
    _, Y, mask = mdata.load_matrix()
    split = df["split"].to_numpy()
    putat = df["is_putative_decoy"].fillna(False).to_numpy().astype(bool)
    isnp = df["is_np"].fillna(False).to_numpy().astype(bool)

    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]
    if a.no_putative:
        tr = tr[~putat[tr]]
        va = va[~putat[va]]
    if a.ft_mode == "np_subset":
        tr = tr[isnp[tr]]

    sw_col = "np_weight" if a.ft_mode == "np_weight" else None
    pw = mdata.pos_weight_from(Y, mask, tr)
    tr_loader, tr_idx = mdata.make_loader(df, ds, tr, a.batch, shuffle=True, sw_col=sw_col,
                                          drop_last=True)
    va_loader, va_idx = mdata.make_loader(df, ds, va, 256)
    te_loader, te_idx = mdata.make_loader(df, ds, te, 256)

    del ds
    import gc; gc.collect()
    model = models.build(a.arch)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(a.epochs):
        model.train()
        run, nb = 0.0, 0
        for b in tr_loader:
            opt.zero_grad()
            loss = models.masked_bce(model(b), b.y, b.mask, pw,
                                     b.sw if sw_col else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss.detach()); nb += 1
        vl = val_loss(model, va_loader, pw)
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": run / max(nb, 1), "val_loss": vl})
        print(f"[{name}] ep{ep} train {run/max(nb,1):.4f} val {vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[{name}] early stop at ep{ep}", flush=True)
                break

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True); os.makedirs("metrics", exist_ok=True)
    torch.save(best_state, f"models/{name}.pt")
    Pv, Yv, Mv, Iv = run_eval(model, va_loader)
    Pt, Yt, Mt, It = run_eval(model, te_loader)
    np.savez_compressed(f"metrics/{name}_preds.npz", Pv=Pv, Yv=Yv, Mv=Mv, Iv=Iv,
                        Pt=Pt, Yt=Yt, Mt=Mt, It=It)
    json.dump({"name": name, "arch": a.arch, "seed": a.seed, "stage": a.stage,
               "ft_mode": a.ft_mode, "no_putative": a.no_putative,
               "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx)),
               "pos_weight": {t: float(w) for t, w in zip(TASKS, pw)},
               "best_val_loss": best, "epochs_run": len(hist),
               "params": int(sum(p.numel() for p in model.parameters())),
               "wall_s": round(time.time() - t0, 1), "history": hist,
               "hparams": {"batch": a.batch, "lr": a.lr, "wd": a.wd, "epochs_max": a.epochs,
                           "patience": a.patience}},
              open(f"metrics/{name}_run.json", "w"), indent=1)
    print(f"[{name}] DONE best_val {best:.4f} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
