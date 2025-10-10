import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import os

from src.utils.general import store_csv, get_data_path, open_csv 


class PredictionProcessor: 
    def __init__(self, CVmain): 
        self.CVmain = CVmain 

    def _save_probas(
            self, 
            probas,
            ytrue = None,  
    ): 
        """
        Saves probability distributions for both the validation set and the data-to-predict.
        
        Args: 
            probas np.array: 
                (2 x n_fights x n_classes) x (n_repeats x n_folds)-dimensional 
                array with predicted probabilities 
            save_for str: 
                'valid' or 'pred'
            n_classes: 
                For how many classes you want the probability distribution.   
            ytrue array: 
                The true predictions. When binary and save_for = valid, used 
                to remove draws. 
        """
        save_for = 'valid' if ytrue is not None else 'pred'

        name_path = get_data_path('interim') / f'{save_for}_names.csv' 
        df_names = open_csv(name_path)

        if ytrue is not None: 
            df_names['result'] = ytrue

        real_cols = ['Win KO','Win Sub','Win Dec','Draw','Loss Dec','Loss Sub','Loss KO'] 
        cols = ['Aa', 'Ab', 'Ac', 'B', 'Cc', 'Cb', 'Ca'] 

        sample_cols = [str(i) for i in range(probas.shape[-1])] 
        df_prob = pd.DataFrame(columns=sample_cols, data=probas) 
        df_names = df_names.merge(pd.DataFrame({'outcome':cols}), how='cross') 
        df_prob = pd.concat([df_names, df_prob], axis=1)

        df_prob1 = df_prob[df_prob['upp_low']==0].copy().sort_values(
            by=['temp_f_id','outcome'], ascending=[True, False]
        )
        sample_cols2 = [str(i) for i in range(probas.shape[-1], 2*probas.shape[-1])] 
        df_prob2 = df_prob[df_prob['upp_low']==1].copy().sort_values(
            by=['temp_f_id','outcome'], ascending=[True,True]
        )
        df_prob2.rename(columns=dict(zip(sample_cols, sample_cols2)), inplace=True)
        sample_cols.extend(sample_cols2)
        df_prob1.reset_index(drop=True, inplace=True)
        df_prob2.reset_index(drop=True, inplace=True)
        
        # Merge upp_low halves side-by-side so each fight/outcome has both sample sets
        df_prob7 = pd.concat([df_prob1, df_prob2[sample_cols2]], axis=1)

        # Get probability distributions and save (now averages include both halves)
        self._probability_distribution(df_prob7, sample_cols, 7) 

        # 7-class calibration (keep draws included so all 7 outcomes are evaluated)
        if save_for == 'valid': 
            self._plot_calibration(df_prob7, 7)

        # For 2-class, exclude draws from targets
        df_for_binary = df_prob7[df_prob7['result'] != 3] if save_for == 'valid' else df_prob7

        # Group outcomes -> Win (Aa,Ab,Ac), Draw (B), Loss (Cc,Cb,Ca)
        win_labels = ['Aa', 'Ab', 'Ac']
        draw_label = 'B'
        loss_labels = ['Cc', 'Cb', 'Ca']

        gcols = ['name f1', 'name f2']
        num_cols = sample_cols  # both sample sets for each row

        wins_sum = (
            df_for_binary[df_for_binary['outcome'].isin(win_labels)]
            .groupby(gcols, as_index=False)[num_cols].sum()
        )
        losses_sum = (
            df_for_binary[df_for_binary['outcome'].isin(loss_labels)]
            .groupby(gcols, as_index=False)[num_cols].sum()
        )
        draws_sum = (
            df_for_binary[df_for_binary['outcome'] == draw_label]
            .groupby(gcols, as_index=False)[num_cols].sum()
        )

        def _adjust_by_draw(df_sum, draws_df):
            merged = df_sum.merge(draws_df, on=gcols, how='left', suffixes=('', '_draw'))
            for c in num_cols:
                denom = 1 - merged[f'{c}_draw'].fillna(0)
                denom = denom.replace(0, 1e-9)
                merged[c] = merged[c] / denom
            return merged[gcols + num_cols]

        wins_adj = _adjust_by_draw(wins_sum, draws_sum)
        losses_adj = _adjust_by_draw(losses_sum, draws_sum)

        wins_final = wins_adj.copy()
        wins_final['outcome'] = 'Win'
        losses_final = losses_adj.copy()
        losses_final['outcome'] = 'Lose'

        if save_for == 'valid' and 'result' in df_for_binary.columns:
            fight_result = df_for_binary[['name f1', 'name f2', 'result']].drop_duplicates()
            wins_final = wins_final.merge(fight_result, on=['name f1', 'name f2'], how='left')
            losses_final = losses_final.merge(fight_result, on=['name f1', 'name f2'], how='left')

        df_prob2 = pd.concat([wins_final, losses_final], ignore_index=True)

        self._probability_distribution(df_prob2, sample_cols, 2) 

        if save_for == 'valid': 
            self._plot_calibration(df_prob2, 2) 

            



    def _probability_distribution(self, df_prob, sample_cols, n_classes): 
        # Extract samples matrix (rows = name/outcome rows, cols = repeats/folds)
        probas = df_prob[sample_cols].values

        avg = np.mean(probas, axis=1)
        std = np.std(probas, axis=1) 
        perc5 = np.percentile(probas, q=5, axis=1)
        perc95 = np.percentile(probas, q=95, axis=1)
        mean2stdp = avg + 2*std 
        mean2stdm = avg - 2*std
        minp = np.min(probas,axis=1) 
        maxp = np.max(probas,axis=1) 
        
        df_preds = pd.DataFrame(
            columns=['avg','std','perc5','perc95','mean2stdp','mean2stdm','min','max'],
            data=np.array([avg, std, perc5, perc95, mean2stdp, mean2stdm, minp, maxp]).T,
        )

        # Attach metadata from the same (post-processed) df_prob
        df_preds = pd.concat([df_prob[['name f1', 'name f2','outcome']], df_preds], axis=1)

        save_for = 'valid' if 'result' in df_prob.columns else 'pred'
        preds_path = self.CVmain.cal_preds_path if save_for == 'valid' else self.CVmain.preds_path
        preds_path = str(preds_path).replace('preds', f'preds{n_classes}') 
        store_csv(df_preds, preds_path)
        logger.info(f'Stored predictions on {save_for} for {n_classes}') 

    def _plot_calibration(
            self, 
            valid_probas, 
            n_classes
    ): 
        # Wilson CI helper with min-count filter and fixed bin centers
        def _bin_stats(conf, target, bins, min_count=25, z=1.96):
            conf = np.asarray(conf, float)
            target = np.asarray(target, bool)

            # Fixed centers and per-bin masks
            centers = (bins[:-1] + bins[1:]) / 2.0
            counts, _ = np.histogram(conf, bins=bins)

            # Compute per-bin correct counts
            correct = np.zeros_like(counts, dtype=float)
            for i in range(len(bins) - 1):
                left, right = bins[i], bins[i + 1]
                if i < len(bins) - 2:
                    mask = (conf >= left) & (conf < right)
                else:
                    mask = (conf >= left) & (conf <= right)
                if mask.any():
                    correct[i] = target[mask].sum()

            n = counts.astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                phat = np.divide(correct, n, out=np.zeros_like(n), where=n > 0)

                denom = 1.0 + (z*z)/n
                center = (phat + (z*z)/(2.0*n)) / denom
                rad = z * np.sqrt((phat*(1.0 - phat) + (z*z)/(4.0*n)) / n, where=n > 0)
                rad = np.divide(rad, denom, out=np.zeros_like(n), where=n > 0)

                lower = np.clip(center - rad, 0.0, 1.0)
                upper = np.clip(center + rad, 0.0, 1.0)

            valid_mask = (n >= min_count)
            err_low = center - lower
            err_high = upper - center
            return centers, center, (err_low, err_high), counts, valid_mask

        # Collect sample columns
        sample_cols = [c for c in valid_probas.columns if isinstance(c, str) and c.isdigit()]
        if not sample_cols:
            logger.warning("No sample columns detected for calibration.")
            return

        suffix = self.CVmain.suffix
        plot_path = str(self.CVmain.cal_plot_path).replace(suffix, f'{suffix}{n_classes}')

        if n_classes == 7:
            # Map outcomes to result codes and display names (exclude Draw -> 2x3 grid)
            labels = ['Aa','Ab','Ac','Cc','Cb','Ca']
            label_to_code = {'Aa':0, 'Ab':1, 'Ac':2, 'Cc':4, 'Cb':5, 'Ca':6}
            label_to_name = {
                'Aa':'Win KO','Ab':'Win Sub','Ac':'Win Dec',
                'Cc':'Loss Dec','Cb':'Loss Sub','Ca':'Loss KO'
            }

            # Subplots grid (2x3)
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            axes = axes.flatten()

            for i, lab in enumerate(labels):
                ax = axes[i]
                ax2 = ax.twinx()

                df = valid_probas[valid_probas['outcome'] == lab].copy()
                if df.empty or 'result' not in df.columns:
                    ax.set_title(f"{label_to_name[lab]} (no data)")
                    ax.set_axis_off()
                    continue

                sample_cols = [c for c in valid_probas.columns if isinstance(c, str) and c.isdigit()]
                conf_m = df[sample_cols].values
                conf = conf_m.reshape(-1)
                target_row = (df['result'] == label_to_code[lab]).values
                target = np.repeat(target_row, conf_m.shape[1])

                if conf.size == 0:
                    ax.set_title(f"{label_to_name[lab]} (no conf)")
                    ax.set_axis_off()
                    continue

                max_conf = float(np.nanmax(conf)) if conf.size else 1.0
                max_conf = min(max(max_conf, 0.02), 1.0)
                bins = np.arange(0.0, max_conf + 1e-9, 0.02)
                if len(bins) < 2:
                    bins = np.array([0.0, max_conf])

                centers, accs, (err_lo, err_hi), counts, valid_mask = _bin_stats(conf, target, bins)

                if valid_mask.any():
                    ax.errorbar(
                        centers[valid_mask], accs[valid_mask],
                        yerr=[err_lo[valid_mask], err_hi[valid_mask]],
                        fmt='o', ms=3, capsize=2, elinewidth=0.8,
                        color='tab:blue', clip_on=True
                    )
                    bin_width = (bins[1] - bins[0]) if len(bins) > 1 else 0.02
                    x_left = max(bins[0], centers[valid_mask].min() - 0.5*bin_width)
                    x_right = min(bins[-1], centers[valid_mask].max() + 0.5*bin_width)
                    ax.set_xlim(x_left, x_right)
                    ax2.set_xlim(x_left, x_right)
                else:
                    ax.set_xlim(bins[0], bins[-1])
                    ax2.set_xlim(bins[0], bins[-1])

                bin_centers = (bins[:-1] + bins[1:]) / 2.0
                ax2.bar(
                    bin_centers, counts,
                    width=(bins[1]-bins[0]) * 0.9 if len(bins) > 1 else 0.018,
                    alpha=0.15, color='tab:gray'
                )

                ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
                ax.set_ylim(0.0, 1.0)
                ax.set_title(label_to_name[lab])
                if i // 3 == 1:
                    ax.set_xlabel('Confidence')
                if i % 3 == 0:
                    ax.set_ylabel('Accuracy')
                ax2.set_ylabel('# Samples')

            fig.tight_layout()
            try:
                if os.path.exists(plot_path):
                    os.remove(plot_path)
                plt.savefig(plot_path, dpi=150)
                logger.info(f"Stored calibration plot at: {plot_path}")
            finally:
                plt.close(fig)
            return

        # n_classes == 2 (single-plot)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax2 = ax.twinx()

        # Use only Win outcome; p_win >= 0.5 corresponds to predicted Win
        df = valid_probas[valid_probas['outcome'] == 'Win'].copy()
        if df.empty or 'result' not in df.columns:
            logger.warning("No valid rows/results for 2-class calibration.")
            plt.close(fig); return

        conf_m = df[sample_cols].values
        conf = conf_m.reshape(-1)
        target_row = df['result'].isin([0, 1, 2]).values  # true if fight was a win
        target = np.repeat(target_row, conf_m.shape[1])

        # Prepare bins on [0.5, 1]
        mask = conf >= 0.5
        conf_f = conf[mask]
        target_f = target[mask]
        if conf_f.size == 0:
            logger.warning("No samples with confidence >= 0.5 for 2-class calibration.")
            plt.close(fig); return

        bins = np.arange(0.5, 1.0001, 0.02)
        centers, accs, (err_lo, err_hi), counts, valid_mask = _bin_stats(conf_f, target_f, bins)

        if valid_mask.any():
            ax.errorbar(
                centers[valid_mask], accs[valid_mask], yerr=[err_lo[valid_mask], err_hi[valid_mask]],
                fmt='o', ms=3, capsize=2, elinewidth=0.8, label='Win', color='tab:blue', clip_on=True
            )
            # Dynamic x-limits to where we actually have valid bins
            bin_width = (bins[1] - bins[0]) if len(bins) > 1 else 0.02
            x_left = max(0.5, centers[valid_mask].min() - 0.5*bin_width)
            x_right = min(1.0, centers[valid_mask].max() + 0.5*bin_width)
            ax.set_xlim(x_left, x_right)
            ax2.set_xlim(x_left, x_right)
        else:
            bin_width = (bins[1] - bins[0]) if len(bins) > 1 else 0.02
            ax.set_xlim(0.5, 1.0); ax2.set_xlim(0.5, 1.0)

        # Distribution bar plot using histogram for bin counts
        hist_counts, _ = np.histogram(conf_f, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        ax2.bar(bin_centers, hist_counts, width=(bins[1]-bins[0]) * 0.9 if len(bins) > 1 else 0.018, alpha=0.2, color='tab:gray')

        ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', lw=1, alpha=0.6)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Confidence (p_win)')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('# Samples')
        ax.set_title('Calibration curve (Win)')
        ax.legend(loc='lower right')

        fig.tight_layout()
        try:
            if os.path.exists(plot_path):
                os.remove(plot_path)
            plt.savefig(plot_path, dpi=150)
            logger.info(f"Stored calibration plot at: {plot_path}")
        finally:
            plt.close(fig)




