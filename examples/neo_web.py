from openchat import OpenChats
from openchat import OpenChat

if __name__ == '__main__':
    OpenChats(models=["gptneo.xlarge"], device="cuda", environment="webserver")
