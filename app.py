from json.tool import main
import multiprocessing
import subprocess
from tkinter import *
import tkinter as tk
import tkinter.messagebox as mbox
from tkinter import filedialog
from PIL import ImageTk, Image

# Main Window & Configuration
window = tk.Tk()
window.title("Real Time CCTV Monitoring")
window.iconbitmap('Images/icon1.ico')
window.geometry('400x180')

# top label
start1 = tk.Label(text = "Real Time CCTV Monitoring System", font=("Calibri", 18), fg="black") # same way bg
start1.place(x = 20, y = 10)

def start_run():
    window.destroy()


# created a start button
Button(window, text="▶ START",command=start_run,font=("Calibri", 14), bg = "white", fg = "black", cursor="hand2", borderwidth=0, relief="raised").place(x =70 , y =100 )

# image on the main window
path1 = "Images/nynoxlogo1.png"
img1 = ImageTk.PhotoImage(Image.open(path1))
panel1 = tk.Label(window, image = img1)
panel1.place(x = 150, y = 50)

exit1 = False
# function created for exiting from window
def exit_win():
    global exit1
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        exit1 = True
        window.destroy()

# exit button created
Button(window, text="❌ EXIT",command=exit_win,font=("Calibri", 14), bg = "white", fg = "black", cursor="hand2", borderwidth=0, relief="raised").place(x =240 , y = 100 )

window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()

if exit1==False:
    subprocess.run(["C:\\Users\\asus\\Documents\\Pythons\\People_counting\\people-counting-main\\myenv\\Scripts\\python", "main.py"])
    exit1 = True
    window.destroy()
