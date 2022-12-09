from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from podm.metrics import MethodAveragePrecision, MetricPerClass


def plot_precision_recall_curve(result: MetricPerClass,
                                dest,
                                lrs_class,
                                method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation,
                                show_ap: bool=False,
                                show_interpolated_precision: bool=False,
                                ):
    """PlotPrecisionRecallCurve
    Plot the Precision x Recall curve for a given class.
    Args:
        result: metric per class
        dest: the plot will be saved as an image in this path
        method: method for interpolation
        show_ap: if True, the average precision value will be shown in the title of
            the graph (default = False);
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    mpre = result.interpolated_precision
    mrec = result.interpolated_recall

    plt.close()
    if show_interpolated_precision:
        if method == MethodAveragePrecision.AllPointsInterpolation:
            plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
        elif method == MethodAveragePrecision.ElevenPointsInterpolation:
            # Uncomment the line below if you want to plot the area
            # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
            # Remove duplicates, getting only the highest precision of each recall value
            nrec = []
            nprec = []
            for idx in range(len(mrec)):
                r = mrec[idx]
                if r not in nrec:
                    idxEq = np.argwhere(mrec == r)
                    nrec.append(r)
                    nprec.append(max([mpre[int(id)] for id in idxEq]))
            plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
    plt.plot(result.recall, result.precision, '-')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1].tolist() + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mpre[:-1].tolist() + [0.0] + [mpre[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    
    #plot the plots at percentage threshold location
    print(result.label)
    for index, percentage in lrs_class:
        plt.plot(result.recall[index],result.precision[index],'mo')
        label = f"{percentage:.2f}"
        label_p = f"confidence:{percentage:.4f} recall:{result.recall[index]:.4f} precision:{result.precision[index]:.4f}"
        print(label_p)
        plt.annotate(label, # this is the text
                    (result.recall[index],result.precision[index]), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(12,6), # distance from text to points (x,y)
                    fontsize=8,
                    ha='center') # horizontal alignment can be left, right or center
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if show_ap:
        ap_str = "{0:.2f}%".format(result.ap * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (result.label, ap_str))
    else:
        plt.title('Precision x Recall curve \nClass: %s' % result.label)
    # plt.legend(shadow=True)
    plt.grid()
    plt.savefig(str(dest))


def plot_precision_recall_curve_all(results: Dict[Any, MetricPerClass],
                                    dest_dir,
                                    lrs_list,
                                    method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation,
                                    show_ap: bool=False,
                                    show_interpolated_precision: bool=False,
                                    ):
    """
    Plot the Precision x Recall curve for a given class.

    Args:
        results: metric per class
        dest_dir: the plot will be saved as an image in this path
        method: method for interpolation
        show_ap: if True, the average precision value will be shown in the title of
            the graph (default = False);
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    for label, result in results.items():
        dest = str(dest_dir +'/'+ label + '_pr.png')
        try:
            plot_precision_recall_curve(result, dest,lrs_list[label], method, show_ap, show_interpolated_precision)
        except:
            print(f'{label}: Cannot plot')

def plot_precision_recall_curve_comb(results: Dict[Any, MetricPerClass],
                                dest_dir,
                                method: MethodAveragePrecision = MethodAveragePrecision.AllPointsInterpolation,
                                show_interpolated_precision: bool=False,
                                ):
    """PlotPrecisionRecallCurve
    Plot the Precision x Recall curve for a given class.
    Args:
        results: metric per class
        dest_dir: the plot will be saved as an image in this path
        method: method for interpolation
        show_interpolated_precision (optional): if True, it will show in the plot the interpolated
            precision (default = False);
    """
    plt.close()
    dest = str(dest_dir +'/all_class_pr.png')
    results['All'] = get_mean_results(results)
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#17becf', '#e377c2', '#7f7f7f', '#bcbd22', '#8c564b']
    for i, (label, result) in enumerate(results.items()):
        labels.append(label)

        mpre = result.interpolated_precision
        mrec = result.interpolated_recall


        if show_interpolated_precision:
            if method == MethodAveragePrecision.AllPointsInterpolation:
                plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
            elif method == MethodAveragePrecision.ElevenPointsInterpolation:
                # Uncomment the line below if you want to plot the area
                # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                # Remove duplicates, getting only the highest precision of each recall value
                nrec = []
                nprec = []
                for idx in range(len(mrec)):
                    r = mrec[idx]
                    if r not in nrec:
                        idxEq = np.argwhere(mrec == r)
                        nrec.append(r)
                        nprec.append(max([mpre[int(id)] for id in idxEq]))
                plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
        plt.plot(result.recall, result.precision, '-', color = colors[i],label = f"{label} mAP:{result.ap:.2f}" )
        # add a new penultimate point to the list (mrec[-2], 0.0)
        # since the last line segment (and respective area) do not affect the AP value
        area_under_curve_x = mrec[:-1].tolist() + [mrec[-2]] + [mrec[-1]]
        area_under_curve_y = mpre[:-1].tolist() + [0.0] + [mpre[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.05, edgecolor=colors[i])

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.xlabel('Recall')
        plt.ylabel('Precision')


    ap_str = "{0:.2f}%".format(results['All'].ap * 100)
    plt.title('Precision x Recall curve for all classes \nmAP: %s' % ap_str)
    plt.legend(shadow=True,loc="lower left")
    plt.grid()
    plt.savefig(str(dest))

def get_mean_results(results:  Dict[Any, MetricPerClass]):
    
    results_all = {}
    for i , values in enumerate(results.values()):
        if i == 0:
            results_all['label'] = 'All'
            results_all['precision'] = values.precision
            results_all['recall'] = values.recall
            results_all['ap'] = values.ap*values.num_detection
            results_all['interpolated_precision'] = values.interpolated_precision
            results_all['interpolated_recall'] = values.interpolated_recall
            results_all['num_groundtruth'] = values.num_groundtruth
            results_all['num_detection'] = values.num_detection
            results_all['tp'] = values.tp
            results_all['fp'] = values.fp
        results_all['precision'] = np.append(results_all['precision'],values.precision)
        results_all['recall'] = np.append(results_all['recall'],values.recall)
        results_all['ap'] += values.ap*values.num_detection
        results_all['interpolated_precision'] = np.append(results_all['interpolated_precision'], values.interpolated_precision)
        results_all['interpolated_recall'] = np.append(results_all['interpolated_recall'], values.interpolated_recall)
        results_all['num_groundtruth'] += values.num_groundtruth
        results_all['num_detection'] += values.num_detection
        results_all['tp'] += values.tp
        results_all['fp'] += values.fp
    results_all['precision'] = np.sort(results_all['precision'])[::-1]
    results_all['recall'] = np.sort(results_all['recall'])
    results_all['ap'] /= results_all['num_detection']
    results_all['interpolated_precision'] = np.sort(results_all['interpolated_precision'])[::-1]
    results_all['interpolated_recall'] = np.sort(results_all['interpolated_recall'])
    
    r = MetricPerClass()
    r.label = results_all['label']
    r.precision = results_all['precision']
    r.recall = results_all['recall']
    r.ap = results_all['ap']
    r.interpolated_precision = results_all['interpolated_precision']
    r.interpolated_recall = results_all['interpolated_recall']
    r.tp = results_all['tp']
    r.fp = results_all['fp']
    r.num_groundtruth = results_all['num_groundtruth']
    r.num_detection = results_all['num_detection']
    
    return r