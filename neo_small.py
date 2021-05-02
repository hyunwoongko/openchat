from openchat import OpenChats
from openchat import OpenChat

if __name__ == '__main__':
    OpenChats(models=["gptneo.small"], device="cpu", environment="webserver")
    #OpenChat(model="gptneo.small", device="cpu", environment="interactive")