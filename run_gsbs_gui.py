# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 06:18:49 2022

@author: dorag
"""

'''
Needed Sections:
    Load data section:
        path text input
        load button to load in memory
        
    Raw data visualizer:
        ask the user to load searchlight/roi data and plot voxel by tr matrix using imshow
    Correlation Matrix visualizer:
        imshow the correlation matrix of the data
        
    avobe 2 can be plotted as data is loaded, load data button can be load and plot
    
    GSBS running section:
        parameter selections with recommendations
    GSBS work in progress plots:
        plot boundary locations like lindas presentations
        
    Results visualizer
        Plot t-dist curve, make people select a solution and plot the bounds on correlation matrix, as well as bounds on timepoints
        
        plo state timeseries for a given solution too 
        
        consider adding stimuli input to show before after stimuli pairs for example for movie frames
'''

#Import the required libraries
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from tkinter.ttk import Style
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

from statesegmentation import gsbs
#Create an instance of Tkinter Frame
win = Tk()


#Set the geometry
win.geometry("1000x1000")
#win.configure(bg='black')


style = Style()
#style.configure('black.TButton', background='black')

def get_state_patterns_in_timeseries():
    state_patterns = gsbs_object.get_state_patterns(k=w1.get())
    
    bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
    state_timeseries = np.zeros_like(gsbs_object.x)
    for state_count,state_pattern in enumerate(state_patterns):
        if state_count == 0:
            state_timeseries[0:bounds_b[state_count]] = np.swapaxes(np.repeat(state_patterns[state_count][:,  np.newaxis], bounds_b[state_count], axis=1),0,1)
        elif state_count == state_patterns.shape[0]-1:
            state_timeseries[bounds_b[state_count-1]:] = np.swapaxes(np.repeat(state_patterns[state_count][:,  np.newaxis], gsbs_object.x.shape[0] - bounds_b[state_count-1], axis=1),0,1)
        else:
            state_timeseries[bounds_b[state_count-1]:bounds_b[state_count]] = np.swapaxes(np.repeat(state_patterns[state_count][:,  np.newaxis], bounds_b[state_count]-bounds_b[state_count-1], axis=1),0,1)
    return state_timeseries
    
    
    
def load_before_after_stimulus():
    
    stimuli_path = load_stimuli_path_entry.get()
    bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
    
    boundary_dropdown.set_menu(*bounds_b)
    global fig5, ax5,canvas5
    # Voxel Timeseries figure and canvas
    fig5, ax5 = plt.subplots(1)
    ax5.axis('off')
    fig5.tight_layout()
    fig5.suptitle('Before')

    #ax2.get_shared_x_axes().join(ax2s[1,0])
    canvas5 = FigureCanvasTkAgg(fig5, frame2)
    canvas5.get_tk_widget().place(x=440,y=650,width=250,height=150)
    
    global fig6, ax6,canvas6
    # Voxel Timeseries figure and canvas
    fig6, ax6 = plt.subplots(1)
    ax6.axis('off')
    fig6.tight_layout()
    fig6.suptitle('After')
    
    #ax2.get_shared_x_axes().join(ax2s[1,0])
    canvas6 = FigureCanvasTkAgg(fig6, frame2)
    canvas6.get_tk_widget().place(x=700,y=650,width=250,height=150)
    
    
    
def show_before_after_stimulus_for_border():    
    
    ax5.clear()
    ax6.clear()
    
    stimuli_path = load_stimuli_path_entry.get()
    bound_frame_number = int(boundary_dropdown_variable.get())-1+3
    
    prev_stim_path = stimuli_path + str(bound_frame_number-1)+'.jpg'
    next_stim_path = stimuli_path + str(bound_frame_number+1)+'.jpg'
    
    prev_stim = plt.imread(prev_stim_path)
    next_stim = plt.imread(next_stim_path)
    
    ax5.imshow(prev_stim)
    ax6.imshow(next_stim)
    ax6.axis('off')
    ax5.axis('off')
    canvas5.draw()
    canvas6.draw()
    
    bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
    xcoords = bounds_b
    for xc in xcoords:
        ax3.axvline(x = xc , color = 'pink', label = 'axvline - full height')
        ax4.axvline(x = xc , color = 'pink', label = 'axvline - full height')
    ax3.axvline(x = int(boundary_dropdown_variable.get()) , color = 'r', label = 'axvline - full height')
    ax4.axvline(x = int(boundary_dropdown_variable.get()) , color = 'r', label = 'axvline - full height')
    
    canvas3.draw()
    canvas4.draw()

# Define a function to return the Input data
def load_data():
   # label.config(text= 'Data Loaded' , font= ('Helvetica 13'))
   global loaded_ROI_data
   
   loaded_ROI_data = np.load(load_path_entry.get(),allow_pickle=True)

   # the figure that will contain the plot DORA
   #fig, (ax1,ax2) = plt.subplots(2,1,sharex = True)

    # plotting the graph
   ax3.imshow(loaded_ROI_data.T)
   #ax2s[0,0].set_ylabel('Voxels')
   
   ax2.imshow(np.corrcoef(loaded_ROI_data))   
   
   # ax2.set_aspect('auto')
   # ax2.set_ylabel('Timepoints')
   # ax2.set_xlabel('Timepoints')
   #fig.tight_layout()
   canvas2.draw()
   canvas3.draw()


def load_gsbs_obj():
    
    global gsbs_object
    
    gsbs_object = np.load(load_path_entry.get(),allow_pickle=True).item()
    
    w1.configure(to=kmax_entry.get())
    w1.set(gsbs_object.tdists.argmax())
    
    #clear all axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    #plot bounds on corr matrix
    ax2.set_title('Bounds on Correlation Matrix')
    bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
    time_correlation(ax2, gsbs_object.x, bounds_b) 
    #plot tdist curve
    ax1.plot(gsbs_object.tdists)
    #plot 
    ax3.imshow(gsbs_object.x.T)
    ax4.imshow(get_state_patterns_in_timeseries().T)
    
    
    xcoords = bounds_b
    for xc in xcoords:
        ax3.axvline(x = xc , color = 'r', label = 'axvline - full height')
        ax4.axvline(x = xc , color = 'r', label = 'axvline - full height')
    canvas1.draw()
    canvas2.draw()
    canvas3.draw()
    canvas4.draw()

# Save gsbs object
def save_gsbs():
    np.save(save_path_entry.get(),gsbs_object,allow_pickle=True)
# Define a function to return the Input data

def start_gsbs():
   #label.config(text= 'Running GSBS' , font= ('Helvetica 13'))
   global gsbs_object
   gsbs_object = gsbs.GSBS(kmax=int(kmax_entry.get()),x=loaded_ROI_data,statewise_detection=measureSystem,finetune=(int(finetune_entry.get())))
   gsbs_object.fit(showProgressBar=True)
   # slider to visualize
   
   w1.configure(to=kmax_entry.get())
   w1.set(gsbs_object.tdists.argmax())
   
   #clear all axes
   ax1.clear()
   ax2.clear()
   ax3.clear()
   ax4.clear()
   
   #plot bounds on corr matrix
   ax2.set_title('Bounds on Correlation Matrix')
   bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
   time_correlation(ax2, gsbs_object.x, bounds_b) 
   #plot tdist curve
   ax1.plot(gsbs_object.tdists)
   #plot 
   ax3.imshow(loaded_ROI_data.T)
   ax4.imshow(get_state_patterns_in_timeseries().T)
   
   
   xcoords = bounds_b
   for xc in xcoords:
       ax3.axvline(x = xc , color = 'r', label = 'axvline - full height')
       ax4.axvline(x = xc , color = 'r', label = 'axvline - full height')
   canvas1.draw()
   canvas2.draw()
   canvas3.draw()
   canvas4.draw()
   
###### Plotting Function ##########
def time_correlation(ax, data, bounds):
    # ax: where it is plotted
    # data: 2D matrix, time x voxels
    # GSBS (opt): GSBS object

    # Compute corrcoef
    corr = np.corrcoef(data)

    # Plot the matrix
    ax.imshow(corr)
    ax.imshow(corr, cmap = 'viridis', interpolation='none',vmin=-1,vmax=1) #changed colormap and range
    #ax.set_xlabel('Timepoints')
    ax.set_ylabel('Timepoints')


    for i in range(len(bounds)-1):
        rect = patches.Rectangle(
            (bounds[i], bounds[i]),
            bounds[i + 1] - bounds[i],
            bounds[i + 1] - bounds[i],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)


def update_results(val):
    # clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    
    #plot tdist curve
    ax1.plot(gsbs_object.tdists)
    ax1.axvline(x = w1.get(), color = 'g', label = 'axvline - full height')
    
    #plot bounds on correlation matrix
    bounds_b = np.where(gsbs_object.all_bounds[w1.get()] > 0)[0]
    time_correlation(ax2, gsbs_object.x, bounds_b) 

    # plot bounds on raw and state timeseries
    ax3.imshow(gsbs_object.x.T)
    ax4.imshow(get_state_patterns_in_timeseries().T)
    
    xcoords = bounds_b
    for xc in xcoords:
        ax3.axvline(x = xc , color = 'pink', label = 'axvline - full height')
        ax4.axvline(x = xc , color = 'pink', label = 'axvline - full height')
    canvas1.draw()
    canvas2.draw()
    canvas3.draw()
    canvas4.draw()

# FRAME 2
frame2 = tk.LabelFrame(win, text="GSBS")#,background='black')
frame2.place(rely=0.02, relx=0.02, height=1200, width=1000)

# Load path entry widget
load_path_entry = Entry(frame2, width= 42)
load_path_entry.insert(0, "Z://home//dora//shared//Scripts//preGSBS_run-1_PPA_L_mean.npy")
load_path_entry.place(x=5,y=5, height=25, width=390)

# Load stimulus path entry widget
load_stimuli_path_entry = Entry(frame2, width= 42)
load_stimuli_path_entry.insert(0, '//cnas.ru.nl//wrkgrp//STD-Donders-DCC-Geerligs//Djamari//StudyForrest//stimulus//frames 2sec//Run1//')
load_stimuli_path_entry.place(x=440,y=550, height=25, width=390)

# Load boundary selection dropdown for stim visualization
boundary_dropdown_variable = IntVar(frame2)
OPTIONS = [0]
boundary_dropdown_variable.set(OPTIONS[0])
boundary_dropdown = ttk.OptionMenu(frame2, boundary_dropdown_variable, *OPTIONS)
boundary_dropdown.place(x=550,y=600,height=25)

# Load button
ttk.Button(frame2, text= "Load ROI data", command= load_data, style='black.TButton').place(x=410,y=5,height=25)

# Load gsbs_object button
ttk.Button(frame2, text= "Load GSBS Object", command= load_gsbs_obj, style='black.TButton').place(x=520,y=5,height=25)

# Load stim button
ttk.Button(frame2, text= "Load Stimuli", command= load_before_after_stimulus, style='black.TButton').place(x=855,y=550,height=25)

# Show stim button
ttk.Button(frame2, text= "Show Stimuli", command= show_before_after_stimulus_for_border, style='black.TButton').place(x=600,y=600,height=25)

# kmax label
label= Label(frame2, text="set kmax:",)
label.place(x=5,y=35,width=70)

# kmax entry widget
kmax_entry = Entry(frame2, width= 42)
kmax_entry.insert(0, "10")
kmax_entry.place(x=80,y=35,width=50)

# finetune label
finetune_label= Label(frame2, text="set finetune:",)
finetune_label.place(x=150,y=35,width=90)

# finetune entry widget
finetune_entry = Entry(frame2, width= 42)
finetune_entry.insert(0, "1")
finetune_entry.place(x=240,y=35,width=50)

# statewise checker
measureSystem =BooleanVar()
statewise_check = ttk.Checkbutton(frame2, text='Search Statewise', variable=measureSystem,  onvalue=True, offvalue=False)
statewise_check.place(x=300,y=35,width=120)

# Run GSBS Button
run_gsbs_button = ttk.Button(frame2, text= "Run GSBS", command= start_gsbs, style='black.TButton').place(x=450,y=35)

# Save GSBS Button
save_gsbs_button = ttk.Button(frame2, text= "Save GSBS object", command= save_gsbs, style='black.TButton').place(x=550,y=35)

# save path entry widget
save_path_entry = Entry(frame2, width= 42)
save_path_entry.insert(0, "save_name_gsbs.npy")
save_path_entry.place(x=660,y=35, height=25, width=150)


# Load plot canvas

# Tdist curve figure and canvas
fig1, ax1 = plt.subplots(1)
fig1.suptitle('T-dist curve')
fig1.tight_layout()
ax1.set_xlabel('Number of Boundaries')

canvas1 = FigureCanvasTkAgg(fig1, frame2)
canvas1.get_tk_widget().place(x=440,y=70,width=400,height=400)


# Correlation Matrix figure and canvas
fig2, ax2 = plt.subplots(1)
fig2.tight_layout()
fig2.suptitle('Correlation Matrix')
ax2.set_ylabel('Timepoints')
ax2.set_xticks([])
#ax2.get_shared_x_axes().join(ax2s[1,0])
canvas2 = FigureCanvasTkAgg(fig2, frame2)
canvas2.get_tk_widget().place(x=10,y=70,width=400,height=400)


# Voxel Timeseries figure and canvas
fig3, ax3 = plt.subplots(1)
fig3.tight_layout()
fig3.suptitle('Voxel Timeseries')
ax3.set_ylabel('Voxels')
ax3.set_xticks([])
#ax2.get_shared_x_axes().join(ax2s[1,0])
canvas3 = FigureCanvasTkAgg(fig3, frame2)
canvas3.get_tk_widget().place(x=10,y=450,width=400,height=200)

# State pattern timeseries figure and canvas
fig4, ax4 = plt.subplots(1)
fig4.tight_layout()
fig4.suptitle('Voxel State Activity Timeseries')
ax4.set_ylabel('Voxels')
ax4.set_xlabel('Timepoints')
#ax2.get_shared_x_axes().join(ax2s[1,0])
canvas4 = FigureCanvasTkAgg(fig4, frame2)
canvas4.get_tk_widget().place(x=10,y=620,width=400,height=200)
# state pattern imeseries subplot

# solution explorer scale
w1 = Scale(frame2, from_=0, orient=HORIZONTAL,command=update_results)
w1.place(x=440,y=500,width = 400)

win.mainloop()
