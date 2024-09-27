import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import interpolate
import random
import colorsys
import os


class Plotter():
    def __init__(self):
        pass

    def plot_line_data(self, x_data, y_data_set_1, y_data_set_2 = None, axis_label_fontsize= 28, line_thickness=5, tick_label_fontsize = 24, tick_width = 4, offset1 = 0,
                offset2 = 0, swap_top_1 = False, swap_top_2 = False, label_pad_1 = 15, label_pad_2 = 35, file_name=None,
                x_axis_limits = None, y_axis_limits_1 = None, line_color_1 = (110/255, 136/255, 194/255), line_color_2 = (186/255, 60/255, 145/255),
                y_axis_limits_2 = None, x_tick_labels = None, y_tick_labels_1 = None, y_tick_labels_2 = None,
                x_axis_label = '2$\it{θ}$ (degrees)', y_axis_label_1 = 'Intensity', y_axis_label_2 = '', plot_title = None, point_size = 0, colors = 'constant',
                trace_labels = None):
        """
        Create a customized X-ray diffraction (XRD) line graph with adjustable aesthetics and random even darker pastel line color.
        
        Args:
        x_data (list): The data points for the x-axis, typically 2-theta or angle.
        y_data (list): The intensity data points for the y-axis.
        hide_y_ticks (bool): If True, hide the y-axis ticks and tick labels.
        axis_label_fontsize (int): Font size for the x and y axis labels.
        line_thickness (float): Thickness of the plot line.
        tick_label_fontsize (int): Font size for the tick labels.
        tick_width (float): Thickness of the ticks and the box around the plot.
        line_color (str): Optional; Color of the plot line. If None, a random even darker pastel color is chosen.
        """

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.weight'] = 'bold'

        color_choices = [(255/255, 179/255, 186/255), (221/255, 160/255, 221/255), (189/255, 252/255, 201/255), (255/255, 255/255, 204/255), (255/255, 240/255, 245/255), (176/255, 224/255, 230/255),
                         (255/255, 218/255, 185/255), (230/255, 200/255, 246/255), (135/255, 206/255, 250/255), (137/255, 207/255, 240/255), (255/255, 229/255, 180/255), (255/255, 228/255, 225/255),
                         (152/255, 255/255, 204/255), (230/255, 230/255, 250/255), (255/255, 192/255, 203/255), (255/255, 182/255, 193/255), (197/255, 203/255, 255/255), (255/255, 253/255, 208/255),
                         (188/255, 212/255, 230/255), (255/255, 230/255, 236/255), (196/255, 195/255, 240/255), (225/255, 190/255, 231/255), (228/255, 193/255, 255/255), (119/255, 221/255, 119/255),
                         (169/255, 234/255, 254/255), (182/255, 255/255, 240/255), (119/255, 221/255, 229/255), (202/255, 231/255, 255/255), (255/255, 204/255, 204/255), (174/255, 198/255, 207/255)]
        
        darken_factor = 0.8
        for j in range(len(color_choices)):
            h,l,s = colorsys.rgb_to_hls(color_choices[j][0], color_choices[j][1], color_choices[j][2])
            l = l * darken_factor
            color_choices[j] = colorsys.hls_to_rgb(h, l, s)

        # Set global font details for tick labels
        plt.rc('xtick', labelsize=tick_label_fontsize)  # Set x-tick label size
        plt.rc('ytick', labelsize=tick_label_fontsize)  # Set y-tick label size


        fig, ax1 = plt.subplots(facecolor='w', figsize=(10, 7), dpi = 300)

        if colors == 'constant':
            color_list_1 = [line_color_1] * len(y_data_set_1)  
        elif colors == 'vary':
            color_list_1 = random.sample(color_choices, len(y_data_set_1))
            print(color_list_1)
        else:
            color_list_1 = colors
        
        if trace_labels == None:
            trace_labels = ["filler"] * len(y_data_set_1)
            
        if y_axis_label_2:
            ax2 = ax1.twinx()
            axes = [ax1, ax2]
        
        else:
            axes = [ax1]

        if swap_top_1 == True:
            y_data_set_1.reverse()
        
        if swap_top_2 == True:
            y_data_set_2.reverse()

        counter = 0
        for i in range(len(y_data_set_1)):
            y_data = y_data_set_1[i]
            if point_size > 0:
                ax1.plot(x_data, y_data + counter * offset1, linewidth=line_thickness, color=color_list_1[i], marker='o', markersize = point_size, label = trace_labels[i])
            else:
                ax1.plot(x_data, y_data + counter * offset1, linewidth=line_thickness, color=color_list_1[i], label = trace_labels[i])
            counter += 1
        
        if y_data_set_2:
            for y_data in y_data_set_2:
                if point_size > 0:
                    ax2.plot(x_data, y_data + counter * offset2, linewidth=line_thickness, color=line_color_2, marker='o', markersize = point_size)
                else:
                    ax2.plot(x_data, y_data + counter * offset2, linewidth=line_thickness, color=line_color_2)
                counter += 1

        if x_axis_limits:
            ax1.set_xlim(x_axis_limits)
        
        if y_axis_limits_1:
            ax1.set_ylim(y_axis_limits_1)
        
        if y_axis_limits_2:
            ax2.set_ylim(y_axis_limits_2)
        
        
        if x_tick_labels:
            ax1.set_xticks(x_tick_labels)
        
        if y_tick_labels_1:
            ax1.set_yticks(y_tick_labels_1)
        
        if y_tick_labels_2:
            ax2.set_yticks(y_tick_labels_1)

        if trace_labels[0] != "filler":
            ax1.legend(loc='best', fontsize=tick_label_fontsize - 2)




        # Make spines and ticks bolder
        for axis in axes:
            for spine in axis.spines.values():
                spine.set_linewidth(tick_width)
            axis.tick_params(which='major', width=tick_width, length=10)
            axis.tick_params(which='minor', width=tick_width, length=5, direction='out')



        ax1.yaxis.tick_left()
        if len(axes) > 1:
            ax2.yaxis.tick_right()

        # Labels
        ax1.set_xlabel(x_axis_label, size = axis_label_fontsize, labelpad = label_pad_1, weight ='bold')
        ax1.set_ylabel(y_axis_label_1, size = axis_label_fontsize,labelpad = label_pad_1, weight ='bold')

        if len(axes) > 1:
            ax2.yaxis.set_label_position("right")
            ax2.set_ylabel(y_axis_label_2, rotation = 270, size = axis_label_fontsize, labelpad = label_pad_2, weight='bold')
        
        if plot_title:
            ax1.set_title(plot_title, fontsize=axis_label_fontsize + 2)


        # Layout and save
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, dpi=300, transparent=True)
        
        else:
            plt.show()
    
    def plot_stackedbar_data(self, y_data_set, axis_label_fontsize= 28, tick_label_fontsize = 24, tick_width = 4,
                label_pad = 15, colorbar_pad = 35,file_name=None, x_axis_limits = None, y_axis_limits = None, colormap = None,
                x_tick_labels = None, y_tick_labels = None,
                x_axis_label = 'Number of In atoms', y_axis_label = 'Energy (eV)', color_bar_label = 'Fraction of Configurations', plot_title = None, bin_size = 0.1,
                x_data_labels = ['numeric'], bar_width = 0.6, colorbar_tickwidth = 2, cumulative = False, normalized = True):
        
        #y_data dimensionality: [  [(value, frequency), (value, frequency), (value, frequency), (value, frequency)], ... (number of x values)  ]

        max_value = max([max([value for value, _ in y_data]) for y_data in y_data_set])
        
        # Set up bins for values (e.g., 0-1, 1-2, 2-3, etc.)
        bins = np.arange(0, max_value + bin_size, bin_size)

        # Loop over the data sets and plot finer regions for values
        binned_data = np.zeros((len(y_data_set), len(bins)-1))

        for i, y_data in enumerate(y_data_set):
            # Get value and frequency data
            values = np.array([value for value, _ in y_data])
            frequencies = np.array([frequency for _, frequency in y_data])
            
            # Stack the bars by subdividing into smaller bins
            bin_frequencies = 0

            for j in range(len(bins) - 1):
                if cumulative != True:
                    bin_frequencies = 0
                for value, frequency in zip(values, frequencies):
                    if bins[j] <= value < bins[j + 1]:
                        bin_frequencies += frequency

                binned_data[i][j] = bin_frequencies









        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.weight'] = 'bold'

        # Set global font details for tick labels
        plt.rc('xtick', labelsize=tick_label_fontsize)  # Set x-tick label size
        plt.rc('ytick', labelsize=tick_label_fontsize)  # Set y-tick label size


        if colormap == None:
            Eric_colors = [[42 / 255, 228 / 255, 255 / 255, 1],
                           [155 / 255, 86 / 255, 197 / 255, 1],
                           [202 / 255, 112 / 255, 183 / 255, 1],
                           [242 / 255, 163 / 255, 143 / 255, 1]]
            colormap = colors.LinearSegmentedColormap.from_list("custom_palette", Eric_colors, N=256)
        
        new_colors = colormap(np.linspace(0, 1, 256))
        new_colors[0] = [1, 1, 1, 1]
        colormap = colors.ListedColormap(new_colors)

        if x_data_labels[0] == 'numeric':
            x_data_labels = range(len(y_data_set))
        
        fig, ax1 = plt.subplots(facecolor='w', figsize=(10, 7), dpi = 300)

        max_frequency = max([max([value for value in data]) for data in binned_data])
        if cumulative or normalized:
            norm = colors.Normalize(vmin=0, vmax = 1)
        
        else:
            norm = colors.Normalize(vmin=0, vmax = max_frequency)

        for i in range(len(binned_data)):
            bottom = 0
            
            if cumulative or normalized == True:
                cumulative_frequencies = sum(binned_data[i])

            for j in range(len(binned_data[i])):
                if cumulative or normalized:
                    color = colormap(norm(binned_data[i][j]/cumulative_frequencies))
                else:
                    color = colormap(norm(binned_data[i][j]))
                ax1.bar(x_data_labels[i], bins[j + 1] - bins[j], bottom=bottom, width=bar_width, color=color, align = 'center')
                bottom += bins[j + 1] - bins[j]

    
        ax1.set_ylabel(y_axis_label, fontsize=axis_label_fontsize, labelpad = label_pad, weight='bold')
        ax1.set_xlabel(x_axis_label, fontsize=axis_label_fontsize, labelpad = label_pad, weight='bold')

        if x_axis_limits:
            ax1.set_xlim(x_axis_limits)
        
        if y_axis_limits:
            ax1.set_ylim(y_axis_limits)
        

        if x_tick_labels:
            ax1.set_xticks(x_tick_labels)
        
        if y_tick_labels:
            ax1.set_yticks(y_tick_labels)


        for spine in ax1.spines.values():
            spine.set_linewidth(tick_width)
        ax1.tick_params(which='major', labelsize=tick_label_fontsize, width=tick_width, length=10)
        ax1.tick_params(which='minor', labelsize=tick_label_fontsize, width=tick_width, length=5, direction='out')


        # Add colorbar to indicate fractional frequency
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Dummy mappable for colorbar
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label(color_bar_label, labelpad = colorbar_pad, rotation = 270, fontsize=axis_label_fontsize, weight='bold')
        cbar.ax.tick_params(width=colorbar_tickwidth)
        cbar.outline.set_linewidth(colorbar_tickwidth)

        # Set plot title
        if plot_title:
            ax1.set_title(plot_title, fontsize=axis_label_fontsize + 2)

        # Tight layout and save/show
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name, dpi=300, transparent=True)
        else:
            plt.show()


    def plot_shared_x_data(self, data_sets, axis_label_fontsize=32, line_thickness=0, tick_label_fontsize=28, tick_width=4,
                        x_axis_limits=None, y_axis_limits_list=None, x_tick_labels=None, y_tick_labels_list=None,
                        x_axis_label='2$\it{θ}$ (degrees)', y_axis_label_list=None, plot_title=None, point_size=0, file_name=None,
                        eye_guide = True):
        """
        Create subplots with shared x-axes where each set of data is plotted on a separate y-axis.
        The top plot will have no y-axis labels but retains the ticks and tick labels.
        
        Args:
        x_data (list): The data points for the x-axis, typically 2-theta or angle.
        y_data_sets (list of lists): A list containing multiple y-data sets for each subplot.
        y_axis_label_list (list of str): Labels for each subplot's y-axis.
        axis_label_fontsize (int): Font size for the x and y axis labels.
        line_thickness (float): Thickness of the plot line.
        tick_label_fontsize (int): Font size for the tick labels.
        tick_width (float): Thickness of the ticks and the box around the plot.
        point_size (int): Size of the markers at each data point, if set > 0.
        """

        #input data format: [  [[x,y],[x,y],...], [[x,y],[x,y],...]  ]

        # Set font and weight globally
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = tick_label_fontsize

        # Ensure the length of y_data_sets and y_axis_label_list match
        if y_axis_label_list is None:
            y_axis_label_list = [''] * len(data_sets)
        if y_axis_limits_list is None:
            y_axis_limits_list = [None] * len(data_sets)
        if y_tick_labels_list is None:
            y_tick_labels_list = [None] * len(data_sets)

        # Set up the subplots with shared x-axes
        fig, axes = plt.subplots(len(data_sets), 1, figsize=(10, 5*len(data_sets)), sharex=True, dpi=300)

        if len(data_sets) == 1:
            axes = [axes]  # Ensure axes is iterable when there's only one subplot


        for i, (ax, xy_data, y_axis_label, y_axis_limits, y_tick_labels) in enumerate(zip(axes, data_sets, y_axis_label_list, y_axis_limits_list, y_tick_labels_list)):
            # Plot each y_data set on its own y-axis
            xy_data = sorted(xy_data, key=lambda x:x[0])
            x_data = [x for x, _ in xy_data]
            y_data = [y for _, y in xy_data]


            if eye_guide:
                guiding_line = interpolate.UnivariateSpline(x_data, y_data, k = 2, s = 0.001)
                guiding_line_domain =  np.linspace(x_data[0], x_data[-1], 300)
                ax.plot(guiding_line_domain, guiding_line(guiding_line_domain), linestyle = '--', color = 'black', linewidth = line_thickness, dashes = (3,4))

            if point_size > 0:
                ax.plot(x_data, y_data, linewidth = 0, marker='o', markersize=point_size)
            else:
                ax.plot(x_data, y_data, linewidth = 0)
            

            # Set axis limits
            if x_axis_limits:
                ax.set_xlim(x_axis_limits)

            if y_axis_limits:
                ax.set_ylim(y_axis_limits)

            if y_tick_labels:
                ax.set_yticks(y_tick_labels)

            # Customize ticks and spines thickness
            for spine in ax.spines.values():
                spine.set_linewidth(tick_width)
            ax.tick_params(which='major', width=tick_width, length=10)
            ax.tick_params(which='minor', width=tick_width, length=5, direction='out')

            # Set y-axis label for each subplot
            ax.set_ylabel(y_axis_label, fontsize=axis_label_fontsize, weight='bold')

        # Set the x-axis label for the bottom subplot
        axes[-1].set_xlabel(x_axis_label, fontsize=axis_label_fontsize, weight='bold')

        # Set plot title if given
        if plot_title:
            fig.suptitle(plot_title, fontsize=axis_label_fontsize + 2, weight='bold')

        # Adjust layout to remove space between subplots
        plt.subplots_adjust(hspace=0)  # Remove space between subplots

        # Save or show the plot
        if file_name:
            plt.savefig(file_name, dpi=300, transparent=True)
        else:
            plt.show()
    
    def plot_custom_xrd_line_graph(self, x_data, experimental, *ticks, residual = None, calculated = None, file_name = "default.pdf"):
        # hide_y_ticks=True, axis_label_fontsize=18, line_thickness=2.5, tick_label_fontsize=16, tick_width=2, offset = 0, swap_top = False, file_name=None):
        """
        Create a customized X-ray diffraction (XRD) line graph with adjustable aesthetics and random even darker pastel line color.
        
        Args:
        x_data (list): The data points for the x-axis, typically 2-theta or angle.
        y_data (list): The intensity data points for the y-axis.
        hide_y_ticks (bool): If True, hide the y-axis ticks and tick labels.
        axis_label_fontsize (int): Font size for the x and y axis labels.
        line_thickness (float): Thickness of the plot line.
        tick_label_fontsize (int): Font size for the tick labels.
        tick_width (float): Thickness of the ticks and the box around the plot.
        line_color (str): Optional; Color of the plot line. If None, a random even darker pastel color is chosen.
        """
        offset = 0.2
        tick_label_fontsize = 16
        tick_width = 2
        hide_y_ticks = True
        axis_label_fontsize = 18
        swap_top = False
        line_thickness = 3.5

        if calculated is not None:
            max_experimental = max([max(experimental), max(calculated)])
            experimental = experimental/max_experimental
            if residual is not None:
                residual = residual / max_experimental
            calculated  = calculated / max_experimental
        
        else:
            max_experimental = max(experimental)
            experimental = experimental/max_experimental
            if residual is not None:
                residual = residual / max_experimental


        experimental = experimental + offset
        if calculated is not None:
            calculated = calculated + offset
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.weight'] = 'bold'
        residual_color = (180/255, 180/255, 180/255)
        experimental_color = (20/255, 20/255, 20/255)
        calculated_color = (137/255, 207/255, 240/255)


        plt.figure(figsize=(7, 6), dpi=300)  # Create a new figure with higher resolution
        
        # Set global font details for tick labels
        plt.rc('xtick', labelsize=tick_label_fontsize)  # Set x-tick label size
        plt.rc('ytick', labelsize=tick_label_fontsize)  # Set y-tick label size
        
        # Plot the line graph with specified line thickness and color
        if residual is not None:
            plt.plot(x_data, residual, linewidth=line_thickness - 0.5, color=residual_color, alpha=1, zorder = 1)
        if calculated is not None:
            plt.plot(x_data, calculated, linewidth = line_thickness, color = calculated_color, zorder = 2)
        plt.scatter(x_data, experimental, s = 2, c=experimental_color, zorder = 3)
        for tick_set in ticks:
            plt.scatter(tick_set[0], [0.15]*len(tick_set), marker = "|", zorder = 4, s = 100, linewidths=2.5, c = (165/255, 84/255, 224/255))
        
        # Set the axis titles with specified font size
        plt.xlabel('2$\it{θ}$ (degrees)', fontsize=axis_label_fontsize, fontweight = 'bold')  # Italicize theta
        plt.ylabel('Intensity', fontsize=axis_label_fontsize, fontweight = 'bold')
        #plt.title('X-ray Diffraction Pattern', fontsize=axis_label_fontsize + 2)  # Title slightly larger
        
        # Add text for the wavelength (lambda) inside the plot, maintaining consistent notation
        plt.text(0.95, 0.95, '$\it{λ}$: 1.5406 Å', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=tick_label_fontsize + 2)
        #plt.text(0.95, 0.95, '$\it{λ}$: 0.406626 Å', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=tick_label_fontsize + 2)
        
        if hide_y_ticks:
            plt.yticks([])  # Hide y-axis ticks
        
        # Customize ticks and the box
        plt.gca().tick_params(width=tick_width)  # Thicken the ticks
        for spine in plt.gca().spines.values():  # Thicken the box
            spine.set_linewidth(tick_width)
        
        plt.grid(False)  # Remove grid for a cleaner look
        plt.xlim(10, 70)
        plt.ylim(-0.2, 1.1 + offset)  # Set y-axis limits from 0 to 1.05
        plt.tight_layout()
        plt.savefig(file_name, dpi = 300, transparent = True)  # Display the plot
        #plt.show()
        plt.close()  # Close the plot to free up memory