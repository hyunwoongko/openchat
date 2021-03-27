import os


class Colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'


def center(text):
    try:
        return text.center(os.get_terminal_size().columns)
    except:
        return text


def cprint(text, color=Colors.RESET, **kwargs):
    print(color + text + Colors.RESET, **kwargs)


def cinput(text, color=Colors.RESET, **kwargs):
    return input(color + text + Colors.RESET, **kwargs)


def draw_openchat():
    logos = [
        """                                                            """,
        """    ____   ____   ______ _   __   ______ __  __ ___   ______""",
        """   / __ \ / __ \ / ____// | / /  / ____// / / //   | /_  __/""",
        """  / / / // /_/ // __/  /  |/ /  / /    / /_/ // /| |  / /   """,
        """ / /_/ // ____// /___ / /|  /  / /___ / __  // ___ | / /    """,
        """ \____//_/    /_____//_/ |_/   \____//_/ /_//_/  |_|/_/     """,
        """                                                            """,
        """                     ... LOADING ...                        """,
        """                                                            """,
    ]

    for line in logos:
        cprint(
            text=center(line),
            color=Colors.CYAN,
        )
