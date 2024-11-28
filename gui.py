import tkinter as tk
import customtkinter as ctk
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
from sequence import sequence


def predict_out():
    epochs = epochs_entry.get()
    lr = lr_entry.get()
    neurons = neurons_per_layer.get()
    layers = num_layer.get()
    layers_list = []
    bias=None
    

    if not epochs or not epochs.isdigit():
        error_label.configure(text="Please enter a valid integer for epochs")
        return

    try:
        lr = float(lr)
    except ValueError:
        error_label.configure(text="Please enter a valid float for the learning rate")
        return

    if not activation_choice.get():
        error_label.configure(text="Please choose an activation function")
        return

    try:
        layers = int(layers)
    except ValueError:
        error_label.configure(text="Please enter a valid integer for the number of layers")
        return

    try:
        neurons = [int(x) for x in neurons.split(",")]  
        if len(neurons) != layers:
            error_label.configure(text=f"Number of layers ({layers}) does not match the neurons list")
            return
        layers_list = neurons
    except ValueError:
        error_label.configure(text="Please enter a valid comma-separated list of integers for neurons per layer")
        return

    error_label.configure(text="")
    seq=sequence(int(layers),int(epochs),float(lr),activation_choice.get(),bias=bias_check.get() == 1)
    seq.build_layers(layers_list)
    test_Acc,train_acc,conf=seq.train()
    test_acc_label.configure(text_color="green", text="test Accuracy :"+str(round(test_Acc,3)))
    train_acc_label.configure(text_color="green", text="train Accuracy :"+str(round(train_acc,3)))
    plot_confusion_matrix(conf)





    



    
def plot_confusion_matrix(confusion):
    conf_matrix = confusion
    fig, ax = plt.subplots(figsize=(5, 4))  
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    
  

    canvas = FigureCanvasTkAgg(fig, master=root)  
    canvas.draw()

    canvas.get_tk_widget().place(x=350, y=150)

root = tk.Tk()
root.configure(bg="#2E2E2E")
root.geometry("720x480")






fig, ax = plt.subplots()
fig.set_size_inches(6, 5)
canvas = FigureCanvasTkAgg(master=root, figure=fig)
# plot = canvas.get_tk_widget()
# plot.place(x=900, y=20)
epochs_entry = ctk.CTkEntry(root, placeholder_text='epochs')
epochs_entry.place(x=10, y=10)

lr_entry = ctk.CTkEntry(root, placeholder_text='learning rate')
lr_entry.place(x=10, y=60)

num_layer = ctk.CTkEntry(root, placeholder_text='#Layers')
num_layer.place(x=200, y=10)

neurons_per_layer = ctk.CTkEntry(root, placeholder_text='neurons/Layer')
neurons_per_layer.place(x=200, y=60)

example = ctk.CTkLabel(root , text='<------ex\n2 layers=> 3,4\n3 layers=> 7,8,6' , text_color='yellow')
example.place(x=340 ,y=70)

activation_choice = tk.StringVar(value="")
sigmoid = ctk.CTkRadioButton(root, text="sigmoid", variable=activation_choice, value="sigmoid")
sigmoid.place(x=10, y=110)
tanh = ctk.CTkRadioButton(root, text="hyperbolic_tangent", variable=activation_choice, value="hyperbolic_tangent")
tanh.place(x=110, y=110)

bias_check = ctk.CTkCheckBox(root, text='Bias')
bias_check.place(x=10, y=160)

predict_btn=ctk.CTkButton(root,text='predict',command=predict_out)
predict_btn.place(x=100,y=240)


error_label = ctk.CTkLabel(root, text_color="red", text="")
error_label.place(x=10, y=210)

test_acc_label = ctk.CTkLabel(root, text_color="green", text="test Accuracy :" , font=("",20))
test_acc_label.place(x=350,y=10)

train_acc_label = ctk.CTkLabel(root, text_color="green", text="train Accuracy :" , font=("",20))
train_acc_label.place(x=350,y=40)


root.mainloop()
