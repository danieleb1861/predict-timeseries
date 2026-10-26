"""PCA and dynamic PCA analysis helpers."""

def run_pca_dpca_analysis(TT, TTtrainZ, TTvalZ, TTtestZ, TTZ, IMGS_DIR, DPCA, PCAplot):
    """Return the fixed feature order and run only the requested diagnostics."""

    pca_cols = ["act_pow_pu", "react_pow_pu",
                "air_temp", "sea_temp",
                "wind_intensity", "sea_state",
                "lever_diff", "pfactor",
                # "man_port", "man_stbd", # binary vars not used in PCA
                "spd_port", "spd_stbd",
                "dP",
                "moving_avg_5", "moving_avg_10", "moving_avg_20",
                "moving_std_5", "moving_std_10", "moving_std_20",
                # "dQ",
                # "moving_avg_5_Q", "moving_avg_10_Q", "moving_avg_20_Q",
                # "moving_std_5_Q", "moving_std_10_Q", "moving_std_20_Q"
                ]
    # Feature selection is fixed and does not require fitting a PCA model.
    if not (DPCA or PCAplot):
        print("dPCA skipped\n\nPCA skipped\n")
        return pca_cols

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from load_forecasting.plots import save_plot, set_pub_style

    set_pub_style(font_family="Times New Roman", fontsize=9)
    if DPCA is True:

        X = TTtrainZ[pca_cols].to_numpy(np.float32)   # (N, Features)
        N, calc_features = X.shape
        Lag = 100  # lag

        # trajectory matrix D: (N-Lag+1, Lag*F)
        D = np.empty((N - Lag + 1, Lag * calc_features), dtype=np.float32)
        for i in range(N - Lag + 1):
            D[i] = X[i:i+Lag].reshape(-1)   # window i..i+Lag-1 flatten

        num_pca_comps = 10
        dpca = PCA(n_components=num_pca_comps, random_state=0)
        Ztr = dpca.fit_transform(D)

        print("Trajectory matrix pca SHAPE is: ", Ztr.shape)
        print()

        load = dpca.components_.reshape(num_pca_comps, Lag, calc_features).astype(np.float32)  # (pc, lag, feature)

        # loadings vs lag for one feature across PCs
        feat_i = 0  # index in pca_cols
        # Linestyles/markers: also for black&white printing
        linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (1, 2)), (0, (3, 1, 1, 1))]
        markers = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]
        plt.figure()
        for pc in range(num_pca_comps):
            plt.plot(
                range(Lag), load[pc, :, feat_i],
                label=f"PC{pc+1} ({100*dpca.explained_variance_ratio_[pc]:.1f}%)",
                linestyle=linestyles[pc % len(linestyles)],
                marker=markers[pc % len(markers)],
                markevery=max(1, load.shape[1] // 8)  # limit to the marker number
                )

        plt.grid(True)
        plt.xlabel("lag")
        plt.ylabel("PC loading vectors")
        plt.title(f"Weights by time lag: {pca_cols[feat_i]}")
        plt.legend(ncol=2, fontsize=8)
        plt.show()

        # Importance (UNWEIGHTED): treat all PCs equally
        imp_feat = (load**2).sum(axis=(0, 1))   # sum over (pc, lag) -> (Features,)
        imp_lag  = (load**2).sum(axis=(0, 2))   # sum over (pc, feature) -> (Lag window size,)

        # # Importance (WEIGHTED): weight PCs by explained variance ratio
        # w = dpca.explained_variance_ratio_.astype(np.float32)  # (K pca variables,)
        # imp_feat_w = ((load**2) * w[:, None, None]).sum(axis=(0, 1))  # (Features,)
        # imp_lag_w  = ((load**2) * w[:, None, None]).sum(axis=(0, 2))  # (Lag window size,)

        # feature importance (bar)
        plt.figure()
        plt.bar(np.arange(calc_features), imp_feat)
        plt.title("DPCA importance per feature")
        plt.xlabel("feature")
        plt.ylabel("energy")
        plt.xticks(np.arange(calc_features), pca_cols, rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # lag importance (line)
        plt.figure()
        plt.plot(range(Lag), imp_lag)
        plt.grid(True)
        plt.title("DPCA lag importance")
        plt.xlabel("lag")
        plt.ylabel("energy")
        plt.show()

        def window_end_axis(data_frame, Lag, time_col="dtime"):
            nwin = len(data_frame) - Lag + 1
            if time_col in data_frame.columns:
                t = pd.to_datetime(data_frame[time_col]).iloc[Lag-1:].to_numpy()
                return t
            # fallback: indice finestra
            return np.arange(nwin)

        t_end = window_end_axis(TTtrainZ, Lag)

        def _k_at(cum, thr):
            idx = np.where(cum >= thr)[0]
            return int(idx[0] + 1) if len(idx) else len(cum)

        # Cumulative explained variance
        expl = 100.0 * dpca.explained_variance_ratio_
        cum  = np.cumsum(expl)
        x = np.arange(1, len(expl) + 1)

        k95 = _k_at(cum, 95.0)
        k99 = _k_at(cum, 99.0)

        print(f"k95={k95} (cum={cum[k95-1]:.2f}%), k99={k99} (cum={cum[k99-1]:.2f}%)")

        plt.figure()
        plt.plot(x, cum, marker="o")
        plt.grid(True)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance (%)")
        plt.title("dPCA: cumulative explained variance")
        plt.axhline(95, linestyle="--", linewidth=0.8)
        plt.axhline(99, linestyle="--", linewidth=0.8)
        plt.xticks(x)
        plt.ylim(0, 101)
        plt.show()

        # PC1–PC2 (scatter) - by time

        Z = Ztr  # (nwin, K)

        # normalised time for color scale
        if np.issubdtype(np.array(t_end).dtype, np.datetime64):
            t_num = (t_end - t_end[0]) / np.timedelta64(1, "s")
        else:
            t_num = np.asarray(t_end, dtype=float)

        t_norm = (t_num - t_num.min()) / (t_num.max() - t_num.min() + 1e-12)

        plt.figure()
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=t_norm, s=8)
        plt.colorbar(sc, label="time (normalised)")
        plt.grid(True)
        plt.xlabel(f"PC1 ({expl[0]:.1f}%)")
        plt.ylabel(f"PC2 ({expl[1]:.1f}%)")
        plt.title("dPCA trajectory (PC1 vs PC2)")
        plt.show()

        # Score (PC1..PCm)

        m = min(3, Z.shape[1])
        plt.figure()
        for i in range(m):
            plt.plot(t_end, Z[:, i], label=f"PC{i+1} ({expl[i]:.1f}%)")
        plt.grid(True)
        plt.xlabel("time" if np.issubdtype(np.array(t_end).dtype, np.datetime64) else "window index")
        plt.ylabel("score")
        plt.title("dPCA scores over time")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residuals

        # D: trajectory matrix (nwin, L*F)
        D_hat = dpca.inverse_transform(Z)          # (nwin, L*F)
        res   = D - D_hat                          # (nwin, L*F)

        rmse = np.sqrt(np.mean(res**2, axis=1))    # RMSE per finestra
        Q    = np.sum(res**2, axis=1)              # SPE / Q statistic

        lam = dpca.explained_variance_             # (K,)
        T2  = np.sum((Z**2) / lam, axis=1)         # T²

        # quantiles
        rmse_thr = np.quantile(rmse, 0.99)
        Q_thr    = np.quantile(Q,    0.99)
        T2_thr   = np.quantile(T2,   0.99)

        # RMSE
        plt.figure()
        plt.plot(t_end, rmse)
        plt.axhline(rmse_thr, linestyle="--", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("time" if np.issubdtype(np.array(t_end).dtype, np.datetime64) else "window index")
        plt.ylabel("RMSE")
        plt.title("Reconstruction error (RMSE) per window")
        plt.tight_layout()
        plt.show()

        # Q / SPE
        plt.figure()
        plt.plot(t_end, Q)
        plt.axhline(Q_thr, linestyle="--", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("time" if np.issubdtype(np.array(t_end).dtype, np.datetime64) else "window index")
        plt.ylabel("Q")
        plt.title("Squared Prediction Error (SPE)")
        plt.tight_layout()
        plt.show()

        # Hotelling
        plt.figure()
        plt.plot(t_end, T2)
        plt.axhline(T2_thr, linestyle="--", linewidth=0.8)
        plt.grid(True)
        plt.xlabel("time" if np.issubdtype(np.array(t_end).dtype, np.datetime64) else "window index")
        plt.ylabel("T²")
        plt.title("Hotelling T²")
        plt.tight_layout()
        plt.show()
    else:
        print("dPCA skipped")
        print()

    # ====================================================
    # %% -- Dimensionality reduction (PCA)
    # ====================================================

    if PCAplot is True:
        # These projections are consumed only by the PCA plots below.
        pcs = 10  # number of principal components to analyse
        pca = PCA(n_components=pcs, random_state=0)

        Ztr = pca.fit_transform(TTtrainZ[pca_cols].to_numpy(dtype=np.float32))

        Zall = pca.transform(TTZ[pca_cols].to_numpy(dtype=np.float32))

    # # ====================================================
    # # %% -- Unsupervised clustering (KMeans)
    # # ====================================================
    # Kcands = range(3,7)
    # sil = []
    # km_models = []
    # for k in Kcands:
    #     km = KMeans(n_clusters=k, n_init=10, random_state=0)
    #     lab = km.fit_predict(Ztr)
    #     sil.append(silhouette_score(Ztr, lab))
    #     km_models.append(km)

    # km = km_models[int(np.argmax(sil))]
    # Kbest = km.n_clusters

    # idx_clust_all = km.predict(Zall)  # fitted km on Train set, predict all

    # TTtrainZ["clust"] = km.predict(Ztr)
    # TTvalZ["clust"]   = km.predict(Zva)
    # TTtestZ["clust"]  = km.predict(Zte)

    # def add_clust_onehot(data_frame, K):
    #     data_frame = data_frame.copy()
    #     oh = pd.get_dummies(data_frame["clust"], prefix="cl")
    #     oh = oh.reindex(columns=[f"cl_{i}" for i in range(K)], fill_value=0)
    #     return pd.concat([data_frame, oh.astype(np.int8)], axis=1)

    # TTtrainZ = add_clust_onehot(TTtrainZ, Kbest)
    # TTvalZ   = add_clust_onehot(TTvalZ, Kbest)
    # TTtestZ  = add_clust_onehot(TTtestZ, Kbest)

    # # ====================================================
    # # %% -- Rule-based labels (approximated operating modes)
    # # ====================================================

    # man_r = (TT["man_port"] == 1) | (TT["man_stbd"] == 1)
    # nav   = ~man_r

    # y_mode = np.zeros(num_samples, dtype=np.uint8)
    # y_mode[man_r.values] = 1
    # y_mode[nav.values]   = 2
    # modeNames = ["manreadiness", "navigation"]

    # ====================================================
    # %% -- PCA plots
    # ====================================================

    if PCAplot is True:
        # Cumulative explained variance
        explained = 100.0 * pca.explained_variance_ratio_
        cum = explained.cumsum()

        k95 = int((cum >= 95).argmax() + 1)
        k99 = int((cum >= 99).argmax() + 1)

        print(f"k for 95% = {k95}  cum% = {cum[k95-1]:.1f}")
        print(f"k for 95% = {k99}  cum% = {cum[k99-1]:.1f}")
        print("PCA loadings to the cumulative explained variance")
        print(" ".join(f"{x:.1f}" for x in explained))
        print()

        y = np.cumsum(explained[:pcs])
        x = np.arange(1, len(y) + 1)

        fig, ax = plt.subplots(figsize=(3.35, 2.4)) # (6.9, 3.2) for two columns, (3.35, 2.4) for one column
        ax.plot(x, y, marker='o')
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title("PCA variance explained")
        ax.set_xticks(x)
        ax.grid(True)

        # 95%
        ax.axhline(95, linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 101)

        plot_name = "PCA_cumsum_variance"
        save_plot(IMGS_DIR, plot_name)

        # PCA biplot (train space)
        def print_top_loadings(pca, pca_cols, num_top=4):
            """
            Print top loadings for first two principal components.
            """
            for comp in range(num_top):
                pc = pca.components_[comp]
                idx = np.argsort(np.abs(pc))[::-1]
                print(f"Top loadings for PC{comp+1}:")
                for i in idx[:5]:
                    print(f"  {pca_cols[i]}: {pc[i]}")
                print()
        print_top_loadings(pca, pca_cols, num_top=4)

        # PC1-PC2_PLOT
        loadings = pca.components_[:2, :].T  # (n_features, 2)
        strength = np.linalg.norm(loadings, axis=1)
        topk = len(pca_cols) # min(8, len(pca_cols))
        idx = np.argsort(strength)[::-1][:topk]

        fig, ax = plt.subplots(figsize=(6.9, 6.0)) # (6.9, 6) for two columns, (3.35, 3) for one column

        # Correlation circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=0.8)
        # arrows
        for i in idx:
            x, y = loadings[i, 0], loadings[i, 1]
            ax.annotate(
                "", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.9, shrinkA=0, shrinkB=0)
            )
            ax.text(x*1.06, y*1.06, pca_cols[i], va="center", ha="center")

        ax.axhline(0, linewidth=0.6)
        ax.axvline(0, linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title("PCA biplot (loadings)")

        plot_name = "pca12_biplot"
        save_plot(IMGS_DIR, plot_name)

        # PC3-PC4_PLOT
        loadings = pca.components_[2:4, :].T  # (n_features, 2)
        strength = np.linalg.norm(loadings, axis=1)
        topk = len(pca_cols) # min(8, len(pca_cols))
        idx = np.argsort(strength)[::-1][:topk]

        fig, ax = plt.subplots(figsize=(6.9, 6.0)) # (6.9, 6) for two columns, (3.35, 3) for one column

        # Correlation circle
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=0.8)
        # arrows
        for i in idx:
            x, y = loadings[i, 0], loadings[i, 1]
            ax.annotate(
                "", xy=(x, y), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.9, shrinkA=0, shrinkB=0)
            )
            ax.text(x*1.06, y*1.06, pca_cols[i], va="center", ha="center")

        ax.axhline(0, linewidth=0.6)
        ax.axvline(0, linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"PC3 ({explained[2]:.1f}%)")
        ax.set_ylabel(f"PC4 ({explained[3]:.1f}%)")
        ax.set_title("PCA biplot (loadings)")

        plot_name = "pca34_biplot"
        save_plot(IMGS_DIR, plot_name)

        # PCA trajectory over time (colorbar by timestamp)
        t = pd.to_datetime(TT["dtime"])
        t_norm = ((t - t.min()) / (t.max() - t.min())).values.astype(float)

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(Zall[:, 0], Zall[:, 1], c=t_norm, s=10)
        plt.colorbar(sc, label="normalised time")
        ax.grid(True)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA trajectory over time")

        plot_name = "data_trajectory"
        save_plot(IMGS_DIR, plot_name)

        # # PCA colored by KMeans clusters
        # plt.figure(figsize=(7, 6))
        # plt.scatter(Zall[:, 0], Zall[:, 1], c=idx_clust_all, s=5, alpha=0.4)
        # plt.grid(True)
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title(f"PCA projection (all data) colored by KMeans (K={Kbest})")

        # plot_name = "clusters"
        # save_plot(IMGS_DIR, plot_name)

        # # PCA colored by operating mode
        # plt.figure(figsize=(7, 6))
        # plt.scatter(Ztr[:, 0], Ztr[:, 1], s=5, alpha=0.2)  # train only

        # plot_name = "PCA_opmode"
        # save_plot(IMGS_DIR, plot_name)
    else:
        print("PCA skipped")
        print()

    # ====================================================

    return pca_cols
