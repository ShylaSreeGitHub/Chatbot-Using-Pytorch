from tkinter import *
from research_chatbot import get_response, bot_name
#tkinter is gui tool, helps to build applications in python.
BG_GRAY = "#33cc00" #green
BG_COLOR = "#eecafa"
TEXT_COLOR = "#007777"
BG_DIV = "#021b20"
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class chatApplication:

    def __init__(self):
        self.window = Tk() #A top level window
        self._setup_main_window() # Main window, a containder to add widgets.

    def run(self):# To run an application
        self.window.mainloop() # mainloop() is called in tkinter module to run an application.

    def _setup_main_window(self):
        self.window.title("Let's Chat") #A title is given here
        self.window.resizable(width=True, height=True) #We can expand and contract the window.
        self.window.configure(width=470, height=550, bg=BG_COLOR)# We customize our widgets here with size and color
        #Layout is created from here.
        # head label is text about the purpose of other widgets.
        head_label = Label(self.window, bg=BG_COLOR, fg="#021b20",
                           text="ScrapT", font=FONT_BOLD,pady=10)
        #relwidth is relative width
        #relheight is relative height
        #relx and rely is a point where the widget must occur
        head_label.place(relwidth=1)

        #tiny divider
        line = Label(self.window, width=450,bg=BG_DIV)
        line.place(relwidth=1,rely=0.07,relheight=0.012)

        # text widget
        self.text_widget = Text(self.window,width=20, height=2, bg=BG_COLOR,
                                fg=TEXT_COLOR, font=FONT,padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=1)
        scrollbar.configure(command=self.text_widget.yview())

        #bottom label
        bottom_label = Label(self.window, bg="#021b20", height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        bottom_label.place(relwidth=1, rely=0.825)
        #focus method is necessary to active the widgets
        #bind method is assigned to an event and invoked automatically.
        #message entry box
        self.msg_entry = Entry(bottom_label, bg="#ffffff", fg="#107dac",
                               font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self._on_enter_pressed)

        #send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20,
                            bg="#c5c1dc", command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.023, relheight=0.02, relwidth=0.22)


    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    # Input message from user is exchanged via insert and get method.
    def _insert_message(self,msg, sender):
        if not msg:
            return
        self.msg_entry.delete(0,END)
        #user input
        sender = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, sender)
        self.text_widget.configure(state=DISABLED)
        #chatbot's response
        bot_response = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, bot_response)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)


if __name__ == "__main__":
    app = chatApplication()
    app.run()