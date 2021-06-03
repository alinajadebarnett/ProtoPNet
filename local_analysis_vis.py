import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from PIL import Image

import numpy as np
import os
import argparse
import re

import shutil

num_rows = 3

def main():
    # get dir
    parser = argparse.ArgumentParser()
    parser.add_argument('-local_analysis_directory', nargs=1, type=str, default='0')
    parser.add_argument('-test_image_directory', nargs=1, type=str, default='/usr/xtmp/ProtoPNet_retrain/CUB_200_2011/datasets/cub200_cropped/test_cropped/')
    args = parser.parse_args()

    source_dir = args.local_analysis_directory[0]
    test_image_dir = args.test_image_directory
    
    # Loading classes based on directory names
    classname_dict = dict()
    for folder in next(os.walk(test_image_dir))[1]:
        classname_dict[int(folder[0:3])-1] = folder[4:]
    # print(classname_dict)

    os.makedirs(os.path.join(source_dir, 'visualizations_of_expl'), exist_ok=True)

    pred, truth = read_local_analysis_log(os.path.join(source_dir + 'local_analysis.log'))

    anno_opts_cen = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')
    anno_opts_sum = dict(xy=(0, -0.1), xycoords='axes fraction',
            va='center', ha='left')

    ### per class expls
    for top_c in range(1, num_rows+1):
        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(28, 12)

        ncols, nrows = 7, num_rows
        spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

        f_axes = []
        for row in range(nrows):
            f_axes.append([])
            for col in range(ncols):
                f_axes[-1].append(fig.add_subplot(spec[row, col]))

        plt.rcParams.update({'font.size': 14})

        for ax_num, ax in enumerate(f_axes[0]):
            if ax_num == 0:
                ax.set_title("Test image", fontdict=None, loc='left', color = "k")
            elif ax_num == 1:
                ax.set_title("Test image activation\nby prototype", fontdict=None, loc='left', color = "k")
            elif ax_num == 2:
                ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
            elif ax_num == 3:
                ax.set_title("Self-activation of\nprototype", fontdict=None, loc='left', color = "k")
            elif ax_num == 4:
                ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
            elif ax_num == 5:
                ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
            elif ax_num == 6:
                ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
            else:
                pass

        plt.rcParams.update({'font.size': 22})

        for ax in [f_axes[r][4] for r in range(nrows)]:
            ax.annotate('x', **anno_opts_symb)

        for ax in [f_axes[r][5] for r in range(nrows)]:
            ax.annotate('=', **anno_opts_symb)

        # get and plot data from source directory

        orig_img = Image.open(os.path.join(source_dir + 'original_img.png'))

        for ax in [f_axes[r][0] for r in range(nrows)]:
            ax.imshow(orig_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

        top_c_dir = os.path.join(source_dir + f'top-{top_c}_class_prototypes')
        for top_p in range(1,num_rows+1):
            #try:
                h_axis = top_p - 1
                # put info in place
                p_info_file = open(os.path.join(top_c_dir, f'top-{top_p}_activated_prototype.txt'), 'r')
                sim_score, cc_dict, class_str, top_cc_str = read_info(p_info_file, classname_dict, per_class=True)
                cc = top_cc_str
                for ax in [f_axes[h_axis][4]]:
                    ax.annotate(sim_score, **anno_opts_cen)
                    ax.set_axis_off()
                #connection_line = p_info_file.readline()
                #cc = connection_line[len('last layer connection: '):-1]
                for ax in [f_axes[h_axis][5]]:
                    ax.annotate(cc + "\n" + class_str, **anno_opts_cen)
                    ax.set_axis_off()
                for ax in [f_axes[h_axis][6]]:
                    tc = float(cc) * float(sim_score)
                    ax.annotate('{0:.3f}'.format(tc) + "\n" + class_str, **anno_opts_cen)
                    ax.set_axis_off()
                p_info_file.close()
                # put images in place
                p_img = Image.open(os.path.join(top_c_dir, f'top-{top_p}_activated_prototype_full_size.png'))
                for ax in [f_axes[h_axis][2]]:
                    ax.imshow(p_img)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                p_act_img = Image.open(os.path.join(top_c_dir, f'top-{top_p}_activated_prototype_self_act.png'))
                for ax in [f_axes[h_axis][3]]:
                    ax.imshow(p_act_img)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                act_img = Image.open(os.path.join(top_c_dir, f'prototype_activation_map_by_top-{top_p}_prototype_normed.png'))
                for ax in [f_axes[h_axis][1]]:
                    ax.imshow(act_img)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
            #except:
                #print("Exception ocurred. This can be caused by having <3 protos for one class.")
        f_axes[2][4].annotate(f"This {classname_dict[int(truth)]} is classified as a {classname_dict[int(pred)]}.", **anno_opts_sum)
        save_loc1 = os.path.join(source_dir, 'visualizations_of_expl') + f'/top-{top_c}_class.png'
        plt.savefig(save_loc1, bbox_inches='tight', pad_inches=0)
        # if top_c==0:
        #     # save predicted class in another place
        #     os.makedirs('./visualizations_of_expl/', exist_ok=True)
        #     save_loc2 = './visualizations_of_expl/' + str(source_dir.replace('/', '__'))[len('__usr__project__xtmp__mammo__alina_saved_models__'):] + f'top-{top_c+1}_class.png'
        #     shutil.copy2(save_loc1, save_loc2)
    
    ###### all classes, one expl
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 4*num_rows)

    ncols, nrows = 7, num_rows
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 14})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation\nby prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of\nprototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 22})

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    # get and plot data from source directory

    orig_img = Image.open(os.path.join(source_dir + 'original_img.png'))

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(orig_img)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    top_p_dir = os.path.join(source_dir + 'most_activated_prototypes')
    for top_p in range(1, num_rows+1):
        h_axis = top_p - 1 
        # put info in place
        p_info_file = open(os.path.join(top_p_dir, f'top-{top_p}_activated_prototype.txt'), 'r')
        sim_score, cc_dict, class_str, top_cc_str = read_info(p_info_file, classname_dict)
        p_info_file.close()
        for ax in [f_axes[h_axis][4]]:
            ax.annotate(sim_score, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[h_axis][5]]:
            ax.annotate(top_cc_str + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        for ax in [f_axes[h_axis][6]]:
            tc = float(top_cc_str) * float(sim_score)
            ax.annotate('{0:.3f}'.format(tc) + "\n" + class_str, **anno_opts_cen)
            ax.set_axis_off()
        # put images in place
        p_img = Image.open(os.path.join(top_p_dir, f'top-{top_p}_activated_prototype_full_size.png'))
        for ax in [f_axes[h_axis][2]]:
            ax.imshow(p_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        p_act_img = Image.open(os.path.join(top_p_dir, f'top-{top_p}_activated_prototype_self_act.png'))
        for ax in [f_axes[h_axis][3]]:
            ax.imshow(p_act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        act_img = Image.open(os.path.join(top_p_dir, f'prototype_activation_map_by_top-{top_p}_prototype_normed.png'))
        for ax in [f_axes[h_axis][1]]:
            ax.imshow(act_img)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    #summary
    f_axes[2][4].annotate(f"This {classname_dict[int(truth)]} is classified as a {classname_dict[int(pred)]}.", **anno_opts_sum)

    save_loc1 = os.path.join(source_dir, 'visualizations_of_expl') + f'/all_class.png'
    plt.savefig(save_loc1, bbox_inches='tight', pad_inches=0)
    os.makedirs('./visualizations_of_expl/', exist_ok=True)
    save_loc2 = './visualizations_of_expl/' + str(source_dir.replace('/', '__'))[len('__usr__project__xtmp__mammo__alina_saved_models__'):] + f'all_class.png'
    shutil.copy2(save_loc1, save_loc2)
    return

def read_local_analysis_log(file_loc):
    log_file = open(file_loc, 'r')
    for _ in range(7):
        _ = log_file.readline()
    pred = log_file.readline()[len("Predicted: "):]
    actual = log_file.readline()[len("Actual: "):]
    log_file.close()
    return pred, actual


def read_info(info_file, classname_dict, per_class=False):
    sim_score_line = info_file.readline()
    connection_line = info_file.readline()
    proto_index_line = info_file.readline()

    sim_score = sim_score_line[len("similarity: "):-1]
    if per_class:
        cc = connection_line[len('last layer connection: '):-1]
    else:
        cc = connection_line[len('last layer connection with predicted class: '):-1]
    #rotation = info_file.readline()
    cc_dict = dict() 
    for i in range(len(classname_dict)):
        cc_line = info_file.readline()
        circ_cc_str = cc_line[len('proto connection to class ' + str(i) + ':tensor('):-(len(", device='cuda:0', grad_fn=<SelectBackward>)")+1)]
        # circ_cc_str = cc_line[len('proto connection to class ' + str(i+1) + ':tensor('):-(len(", grad_fn=<SelectBackward>)")+1)]
        # print(circ_cc_str, i, classname_dict[i], cc_line)
        circ_cc = float(circ_cc_str)
        cc_dict[i] = circ_cc

    # proto connection to class 0:tensor(1.8295e-07, grad_fn=<SelectBackward>)
    class_of_p = max(cc_dict, key=lambda k: cc_dict[k])
    top_cc = cc_dict[class_of_p]

    class_str = classname_dict[class_of_p]
    top_cc_str = str(top_cc)
    return sim_score, cc_dict, class_str.replace('_', ' '), top_cc_str

def test():

    im = Image.open('./visualizations_of_expl/' + 'original_img.png')

    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(28, 12)

    ncols, nrows = 7, num_rows
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)

    f_axes = []
    for row in range(nrows):
        f_axes.append([])
        for col in range(ncols):
            f_axes[-1].append(fig.add_subplot(spec[row, col]))

    plt.rcParams.update({'font.size': 15})

    for ax_num, ax in enumerate(f_axes[0]):
        if ax_num == 0:
            ax.set_title("Test image", fontdict=None, loc='left', color = "k")
        elif ax_num == 1:
            ax.set_title("Test image activation by prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 2:
            ax.set_title("Prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 3:
            ax.set_title("Self-activation of prototype", fontdict=None, loc='left', color = "k")
        elif ax_num == 4:
            ax.set_title("Similarity score", fontdict=None, loc='left', color = "k")
        elif ax_num == 5:
            ax.set_title("Class connection", fontdict=None, loc='left', color = "k")
        elif ax_num == 6:
            ax.set_title("Contribution", fontdict=None, loc='left', color = "k")
        else:
            pass

    plt.rcParams.update({'font.size': 22})

    for ax in [f_axes[r][0] for r in range(nrows)]:
        ax.imshow(im)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])


    anno_opts = dict(xy=(0.4, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    anno_opts_symb = dict(xy=(1, 0.5), xycoords='axes fraction',
                    va='center', ha='center')

    for ax in [f_axes[r][s] for r in range(nrows) for s in range(4,7)]:
        ax.annotate('Number', **anno_opts)
        ax.set_axis_off()

    for ax in [f_axes[r][4] for r in range(nrows)]:
        ax.annotate('x', **anno_opts_symb)

    for ax in [f_axes[r][5] for r in range(nrows)]:
        ax.annotate('=', **anno_opts_symb)

    os.makedirs('./visualizations_of_expl/', exist_ok=True)
    plt.savefig('./visualizations_of_expl/' + 'test.png')

    # Refs: https://stackoverflow.com/questions/40846492/how-to-add-text-to-each-image-using-imagegrid
    # https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib

if __name__ == "__main__":
    main()
