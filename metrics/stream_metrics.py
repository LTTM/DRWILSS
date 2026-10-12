import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, n_old_classes=None):
        super().__init__()
        self.n_old_classes = n_old_classes  # old classes and background
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

    def to_str(self, results, verbose=True):
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec", "Agg",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        if results["Mean IoU Bkg"] == "X":
            ignore.append("Mean IoU Bkg")
            ignore.append("Mean IoU Old")
            ignore.append("Mean IoU New")
        if results["Final Mean IoU Bkg"] == "X":
            ignore.append("Final Mean IoU Bkg")
            ignore.append("Final Mean IoU Dense")
            ignore.append("Final Mean IoU Incr")
            ignore.append("Final Mean IoU All")
        for k, v in results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        if verbose:
            string += 'Class IoU:\n'
            for k, v in results['Class IoU'].items():
                string += "\tclass %d: %s\n"%(k, str(v))

            for i, name in enumerate(['Class IoU', 'Class Acc', 'Class Prec']):
                string += f"{name}:'\t: {results['Agg'][i]}\n"

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, final_test=False, n_dense_classes=None):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + EPS)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        mean_iu_bkg = np.mean(iu[:1]) if self.n_old_classes is not None else "X"
        mean_iu_old = np.mean(iu[1:self.n_old_classes]) if self.n_old_classes is not None else "X"
        mean_iu_new = np.mean(iu[self.n_old_classes:]) if self.n_old_classes is not None else "X"
        # prepare metrics for final performance test
        final_test_mean_iu_bkg = np.mean(iu[:1]) if final_test else "X"
        final_test_mean_iu_dense = np.mean(iu[1:n_dense_classes]) if final_test else "X"
        final_test_mean_iu_incr = np.mean(iu[n_dense_classes:]) if final_test else "X"
        final_test_mean_iu_all = np.mean(iu[1:]) if final_test else "X"
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        return {
                "Total samples":  self.total_samples,
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean Prec": precision_cls,
                # "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Mean IoU Bkg": mean_iu_bkg,
                "Mean IoU Old": mean_iu_old,
                "Mean IoU New": mean_iu_new,
                "Final Mean IoU Bkg": final_test_mean_iu_bkg,
                "Final Mean IoU Dense": final_test_mean_iu_dense,
                "Final Mean IoU Incr": final_test_mean_iu_incr,
                "Final Mean IoU All": final_test_mean_iu_all,
                "Class Acc": cls_acc,
                "Class Prec": cls_prec,
                "Agg": [mean_iu, acc_cls, precision_cls],
                "Confusion Matrix": self.confusion_matrix_to_fig()
            }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1)+0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

