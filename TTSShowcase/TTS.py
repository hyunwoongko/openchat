# Start by importing the win32com package built into python
import win32com.client as wincom
from openchat import OpenChat

if __name__ == '__main__':
    OpenChat(model="dodecathlon.convai2", device="cpu")

#Create Voice Object    
speak = wincom.Dispatch("SAPI.SpVoice")

#Recive text and assign "Text" value then speak text
text = "Python text-to-speech test. using win32com.client"
speak.Speak(text)

#PyWin32 will directly speak the text using the built-in Microsoft speech engine
#I am not sure how to get the ai inputs to assign to text value tho