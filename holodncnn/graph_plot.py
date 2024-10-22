import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
import multiprocessing
import time
from tkinter import *

from .model import *
# from argument import *
# from main_holo_tsf import *

import numpy as np
import datetime


#Create a window
window=Tk()


# Main need arguments to use it
# ex: python main_test.py --output_dir ./PyTorchCheckpoint/
def main():
    """TO UPDATE with new yaml argument file"""
    args = parse()
    max_epochs = args.num_epochs

    if(args.exp_file):
        # exp_file = './PyTorchCheckpoint/experiment_08_02_2021-18:21:44'
        exp_file = args.exp_file
    else:

        # input_dir = './PyTorchCheckpoint/'
        input_dir = args.output_dir
        checkpoint = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("[*] Loading data...")
        while(checkpoint is None):

            try:       #Try to check if there is data in the queue
                exp_file = 'experiment_{}'.format(datetime.datetime.now().strftime("%d_%m_%Y-%H:%M:%S"))
                exp_file = input_dir + exp_file
                checkpoint = torch.load(os.path.join(exp_file, "checkpoint_{:0>5}.pth.tar".format(0)), map_location=device)
            except:
                time.sleep(0)
        print("[*] Load successfully...")


# ===================================
    #Create a queue to share data between process
    q = multiprocessing.Queue()

    # time.sleep(1)
    #Create and start the simulation process
    simulate=multiprocessing.Process(None,simulation,args=(q,exp_file,max_epochs))
    simulate.start()

    #Create the base plot
    plot()

    #Call a function to update the plot when there is new data
    updateplot(q,exp_file,max_epochs)

    window.mainloop()
    print ('Done')
    # ===================================

def plot():    #Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global line,ax,canvas
    fig = matplotlib.figure.Figure()

    ax = fig.add_subplot(1,1,1,title="Evaluation losses Graph",xlabel="Epochs Number",ylabel="Losses")
    canvas = FigureCanvasTkAgg(fig, master=window)
    # ax.xlabel("Epochs")
    # plt.ylabel("losses")
    window.wm_title("Evaluation losses Graph")
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)


def update_checkpoint(exp_file, max):
    checkpoint = None
    # test = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    while(checkpoint is None):

        try:       #Try to check if there is data in the queue
            checkpoint = torch.load(os.path.join(exp_file, "checkpoint_{:0>5}.pth.tar".format(max)), map_location=device)

        except:
            max-=1

    return checkpoint

def updateplot(q,ex_file,max_epochs):
    try:       #Try to check if there is data in the queue
        result=q.get_nowait()

        if result !='Q':

            checkpoint = update_checkpoint(ex_file,max_epochs)

            history = checkpoint['History']

            loss_tab_val = []
            loss_tab_eval = []

            for k,v in (history):
                loss_tab_val = np.append(loss_tab_val,round(k['loss'],6))
                loss_tab_eval = np.append(loss_tab_eval,round(v['loss'],6))


            y1 = np.arange(0,len(loss_tab_val))
            x1 = loss_tab_val
            x2 = loss_tab_eval

            line, = ax.plot(y1, x1,'r--')
            line, = ax.plot(y1, x2,'b')
            ax.draw_artist(line)
            canvas.draw()
            window.after(200,updateplot,q,ex_file,max_epochs)
        else:
             print ('done')
    except:
        print( "empty")
        window.after(200,updateplot,q,ex_file,max_epochs)


def simulation(q,ex_file,max_epochs):

    checkpoint = update_checkpoint(ex_file,max_epochs)

    history = checkpoint['History']

    loss_tab = []
    for k,v in (history):
        loss_tab = np.append(loss_tab,round(k['loss'],6))
    q.put(loss_tab)
    # q.put(loss_tab)
    while(1):
        time.sleep(0.2)
        q.put("not Q")

    q.put('Q')


if __name__ == '__main__':
    main()
