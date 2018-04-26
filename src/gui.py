#import the 'tkinter' module
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename

globalImPath = ""
mainBackground = "#fafafa"
imBackground = "#ffffff"

#create a new window
window = tkinter.Tk()
#set the window background 
window.configure(background=mainBackground)

def OpenFile():
    globalImPath = askopenfilename(initialdir="",
                           filetypes =(("Image file", "*.png"),("All Files","*.*")),
                           title = "Choose a file."
                           )
    
    try:
       print (globalImPath) ##
       ##this is where u process ur shit dawg
    except:
        print("No file exists")


Title = window.title( "Detect ze diabetes")


#Menu Bar

menu = Menu(window)
window.config(menu=menu)

file = Menu(menu)

file.add_command(label = 'Open', command = OpenFile)
file.add_command(label = 'Exit', command = lambda:exit())

menu.add_cascade(label = 'File', menu = file)
menu.configure(background=mainBackground)
file.configure(background=mainBackground)

preproc_image = tkinter.PhotoImage(file="blankold.png") ##set this to a transparent image of the same dimensions as your preproc image
postproc_image = tkinter.PhotoImage(file="blankold.png") ##set this to a transparent image of the same dimensions as your postproc image

pre = tkinter.Label(window, image=preproc_image, background=imBackground)
post= tkinter.Label(window, image=postproc_image, background=imBackground)

pre.pack(side = LEFT)
##maybe add a seperator between the two
post.pack(side = LEFT)

    
def updateGUI(im1,path1,im2,path2,info,newInfo):
    temp1 = tkinter.PhotoImage(file=path1)
    im1.configure(image=temp1)
    temp2 = tkinter.PhotoImage(file=path2)
    im2.configure(image=temp2)
    im1.image = temp1
    im2.image = temp2

    with open('output.txt', 'r') as myfile:
        data=myfile.read()
        info.configure(text=data)

    window.update()

#placeholder for proc output
info = tkinter.Label(window, text="Please choose images to process...", fg="#383a39", bg=mainBackground, font=("Helvetica", 10), padx = 20)
info.pack(side = TOP)
b = Button(text="test if you can update images live",padx = 40, command=lambda :updateGUI(pre,"cat.png", post, "cat.png", info, "output.txt")) ##use the updateGUI function when your processing is done and remove this button
b.pack(side = BOTTOM)
#draw the window, and start the 'application'
window.mainloop()