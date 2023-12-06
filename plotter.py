import numpy as np
import os
from math import floor, ceil
import matplotlib.pyplot as plt

def plotter(x, y, title, x_label, y_label, settings, x_axis_increment='', y_axis_increment='',
	x_tick_positions = [], y_tick_positions = [], color_option = "gradient", color_i = [7/255, 107/255, 255/255],
	color_f = [208/255, 170/255, 96/255], color_list = [], no_x_ticks = False, no_y_ticks = False, x_range = [],
	y_range = [], offset = 0, diffraction_wavelength = 0, trace_labels = [], plot_upside_down = False, normalize = False,
	min_shift = 0, max_shift = 0):
	settings_dict = {
		"SI": {
			"Figsize": (9, 8),
			"x_labels_font_size": 30,
			"x_label_pad": 10,
			"x_tick_labels_font_size": 18,
			"x_tick_labels_pad": 10,
			"y_labels_font_size": 24,
			"y_label_pad": -25,
			"y_label_pad_noSN": 10,
			"y_tick_labels_font_size": 18,
			"y_tick_labels_pad": 5,
			"title_font_size": 40,
			"title_pad": 15,
			"text_font_size": 20,
			"linewidth": 6,
			"linestyle": '-',
			"marker": '',
			"x_tick_length": 8,
			"y_tick_length": 8,
			"border_thickness": 3,
			"move_bottom": 0.12,
			"move_top": 0.9,
			"move_left": 0.18,
			"move_right": 0.95,
			"show_grid": False,
		},
		"main_text": {
			"Figsize": (9, 8),
			"x_labels_font_size": 30,
			"x_label_pad": 10,
			"x_tick_labels_font_size": 24,
			"x_tick_labels_pad": 10,
			"y_labels_font_size": 30,
			"y_label_pad": -25,
			"y_label_pad_noSN": 10,
			"y_tick_labels_font_size": 24,
			"y_tick_labels_pad": 5,
			"title_font_size": 40,
			"title_pad": 15,
			"text_font_size": 20,
			"linewidth": 3,
			"linestyle": '-',
			"marker": '',
			"x_tick_length": 8,
			"y_tick_length": 8,
			"border_thickness": 6,
			"move_bottom": 0.12,
			"move_top": 0.9,
			"move_left": 0.18,
			"move_right": 0.95,
			"show_grid": False,
		}
	}

	number_of_traces = len(y)
	if color_option == "gradient":
		for i in range(number_of_traces):
			red = (1 - i/(number_of_traces - 1)) * color_i[0] + (i/(number_of_traces - 1)) * color_f[0]
			green = (1 - i/(number_of_traces - 1)) * color_i[1] + (i/(number_of_traces - 1)) * color_f[1]
			blue = (1 - i/(number_of_traces - 1)) * color_i[2] + (i/(number_of_traces - 1)) * color_f[2]
			color_list.append([red, green, blue])

	x_decimal_place = 'x'
	y_decimal_place = 'y'

	if normalize:
		for i in range(number_of_traces):
			y[i] = np.array(y[i]) / (max(y[i]) - min(y[i]))

	if plot_upside_down:
		for i in range(number_of_traces)[1:]:
			y[number_of_traces - i - 1] = max(y[-1]) * offset * i + np.array(y[number_of_traces - i - 1])
	else:
		for i in range(number_of_traces)[1:]:
			y[i] = max(y[0]) * offset * i + np.array(y[i])

	settings = settings_dict.get(settings, {})

	if x_axis_increment == '':
		x_axis_increment, x_decimal_place = calculate_axis_increment(x, x_tick_positions, x_range)

	if y_axis_increment == '':
		y_axis_increment, y_decimal_place = calculate_axis_increment(y, y_tick_positions, y_range)

	if x_decimal_place == 'x':
		x_decimal_place = calculate_decimal_place(x_axis_increment)

	if y_decimal_place == 'y':
		y_decimal_place = calculate_decimal_place(y_axis_increment)

	if x_tick_positions:
		x_range = [x_tick_positions[0], x_tick_positions[-1]]
	else:
		x_tick_positions, x_range = generate_tick_positions(x, x_axis_increment, no_x_ticks, x_range)

	if x_tick_positions:
		x_tick_labels = generate_tick_labels(x_tick_positions, x_axis_increment, x_decimal_place)
	else:
		x_tick_labels = []
	
	if y_tick_positions:
		y_range = [y_tick_positions[0], y_tick_positions[-1]]
	else:
		y_tick_positions, y_range = generate_tick_positions(y, y_axis_increment, no_y_ticks, y_range)

	if y_tick_positions:
		y_tick_labels = generate_tick_labels(y_tick_positions, y_axis_increment, y_decimal_place)
	else:
		y_tick_labels = []



	fig, ax = plt.subplots(figsize=settings.get("Figsize", (9, 8)))
	if plot_upside_down:
		for i in range(number_of_traces):
			ax.plot(x[i], y[i], marker=settings.get("marker", ''), linestyle=settings.get("linestyle", '-'), linewidth=settings.get("linewidth", 6), color = color_list[i])
	else:
		for i in range(number_of_traces):
			ax.plot(x[number_of_traces - i - 1], y[number_of_traces - i - 1], marker=settings.get("marker", ''), linestyle=settings.get("linestyle", '-'), linewidth=settings.get("linewidth", 6), color = color_list[number_of_traces - i - 1])

	if y_decimal_place <= 2 and y_decimal_place >= -2:
		settings['y_label_pad'] = settings['y_label_pad_noSN']
	set_plot_labels(ax, x_label, y_label, title, settings)
	set_plot_ticks(ax, x_tick_positions, y_tick_positions, x_tick_labels, y_tick_labels, settings)
	set_plot_limits(ax, x_range[0], x_range[1], y_range[0] - min_shift, y_range[1] + max_shift)
	set_plot_tick_params(ax, settings)
	add_text(ax, diffraction_wavelength, settings.get("text_font_size", 20))
	if len(trace_labels) > 0:
		add_legend(ax, trace_labels)
	set_plot_spines(ax, settings)

	fig.tight_layout()
	plt.grid(settings.get("show_grid", False))
	return fig, ax

def find_min_and_max(x):
	max_data = x[0][0]
	min_data = x[0][0]
	for data_list in x:
		if max(data_list) > max_data:
			max_data = max(data_list)
		if min(data_list) < min_data:
			min_data = min(data_list)
	return min_data, max_data

def calculate_axis_increment(x, tick_positions, data_range):
	if tick_positions == []:
		if len(data_range) == 2:
			min_data = data_range[0]
			max_data = data_range[1]
		else:
			min_data, max_data = find_min_and_max(x)
		axis_increment = (max_data - min_data)/6
	else:
		axis_increment = tick_positions[1] - tick_positions[0]
	decimal_place = calculate_decimal_place(axis_increment)
	axis_increment = round(axis_increment*10**(-decimal_place))*10**decimal_place
	return axis_increment, decimal_place

def calculate_decimal_place(axis_increment):
	if axis_increment < 1 and "e" not in str(axis_increment):
		decimal_place = 0
		for letter in str(axis_increment).split(".")[1]:
			decimal_place = decimal_place + 1
			if letter != "0":
				break
		decimal_place = -decimal_place
	elif "e" in str(axis_increment):
		decimal_place = float(str(axis_increment).split("e")[1])
	else:
		decimal_place = len(str(axis_increment).split(".")[0]) - 1
	return decimal_place

def generate_tick_positions(x, axis_increment, no_ticks, data_range):
	if len(data_range) == 2:
		min_data = data_range[0]
		max_data = data_range[1]
	else:
		min_data, max_data = find_min_and_max(x)
	tick_positions = range(floor(min_data/axis_increment), ceil(max_data/axis_increment) + 1)
	ticks = [position * axis_increment for position in tick_positions]
	if no_ticks:
		ticks = []

	return ticks, [min_data, max_data]

def generate_tick_labels(tick_positions, axis_increment, decimal_place):
	if decimal_place > 2 or decimal_place < -2:
		print(tick_positions)
		tick_labels = [str(position * 10 ** (-decimal_place)).split(".")[0] for position in tick_positions]
		decimal_place = int(decimal_place)
		tick_labels[-1] += f' x 10$^{{\\mathbf{{ {decimal_place} }}}}$'
	else:
		tick_labels = [f"{position:.{abs(decimal_place)}f}" if "." in str(axis_increment) else str(position) for position in tick_positions]
	return tick_labels

def set_plot_labels(ax, x_label, y_label, title, settings):
	ax.set_xlabel(x_label, fontweight='bold', fontsize=settings.get("x_labels_font_size", 24), fontfamily='Arial', labelpad=settings.get("x_label_pad", 10))
	ax.set_ylabel(y_label, fontweight='bold', fontsize=settings.get("y_labels_font_size", 24), fontfamily='Arial', labelpad=settings.get("y_label_pad", -25))
	ax.set_title(title, fontweight='bold', fontsize=settings.get("title_font_size", 40), fontfamily='Arial', pad=settings.get("title_pad", 15), wrap = True)

def set_plot_ticks(ax, x_tick_positions, y_tick_positions, x_tick_labels, y_tick_labels, settings):
	ax.set_xticks(x_tick_positions)
	ax.set_yticks(y_tick_positions)
	ax.set_xticklabels(x_tick_labels, fontname='Arial', fontweight='bold', fontsize=settings.get("x_tick_labels_font_size", 18))
	ax.set_yticklabels(y_tick_labels, fontname='Arial', fontweight='bold', fontsize=settings.get("y_tick_labels_font_size", 18))

def set_plot_limits(ax, x_min, x_max, y_min, y_max):
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)

def set_plot_tick_params(ax, settings):
	ax.tick_params(axis='x', length=settings.get("x_tick_length", 8), width=settings.get("border_thickness", 3), pad=settings.get("x_tick_labels_pad", 10))
	ax.tick_params(axis='y', length=settings.get("y_tick_length", 8), width=settings.get("border_thickness", 3), pad=settings.get("y_tick_labels_pad", 5))

def set_plot_spines(ax, settings):
	for spine in ['top', 'bottom', 'left', 'right']:
		ax.spines[spine].set_linewidth(settings.get("border_thickness", 3))

def add_text(ax, diffraction_wavelength, font_size):
	if diffraction_wavelength:
		ax.text(.03, .95, r"$\lambda$ = " + str(diffraction_wavelength) + " " + chr(197), ha='left', va='top', fontweight='bold', fontsize = font_size, fontfamily='Arial', transform=ax.transAxes)

def add_legend(ax, trace_labels):
	trace_labels.reverse()
	legend_object = ax.legend(trace_labels)
	plt.setp(legend_object.texts, fontweight='bold', fontsize = 20, fontfamily='Arial')

def take_step(file_name):
	return float(file_name.split("_")[-2])

x_list = []
y_list = []

folder_name = "./data/"

file_list = os.listdir(folder_name)
file_list.sort(key=take_step)

for file in file_list:
	data_file = open(folder_name + file, 'r')
	data_file_lines = data_file.readlines()
	data_file.close()
	x = []
	y = []
	for line in data_file_lines:
		if line[0].isnumeric():
			x.append(float(line.split(",")[0]))
			y.append(float(line.split(",")[1]))
	x_list.append(x)
	y_list.append(y)


plotter(x_list, y_list, "", r"2$\theta$ (" + chr(176) + ")", "Intensity", "main_text", no_y_ticks = True, x_range = [2, 18],
	y_range = [], offset = 0.4, color_f = [100/255, 143/255, 255/255], color_i = [220/255, 38/255, 127/255],
	plot_upside_down = True, normalize = True, diffraction_wavelength = 0.31, min_shift = 0.3, max_shift = 0.2)
plt.savefig("Data.png", dpi=400, transparent = False)
plt.close()