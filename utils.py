import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_or_check_directory(input_directory: str = ""):
    if input_directory and isinstance(input_directory,str):
        if not os.path.exists(input_directory):
            try:
                os.makedirs(input_directory)
                return True
            except Exception as E:
                print(f"{bcolors.FAIL}[ERROR] Unable to create `{input_directory}` because of the following error: {E}{bcolors.ENDC}")
                return False
        return True
    else:
        print(f"{bcolors.FAIL}[ERROR] Unable to create `{input_directory}`. Please verify input formal parameter.{bcolors.ENDC}")


def fancy_print(msg: str=""):
    terminal_size = 80
    difference = terminal_size - 4
    if msg:
        print()
        print("#"*terminal_size)
        print(f"##{msg:^{difference}s}##")
        print("#"*terminal_size)
        print()
    else:
        msg = "No message passed"
        print()
        print("#"*terminal_size)
        print(f"##{msg:^{difference}s}##")
        print("#"*terminal_size)
        print()



def is_empty(input_directory: str = "") -> bool:
    
    if input_directory:
        if os.path.exists(input_directory):
            if len(os.listdir(input_directory)):
                return False

    return True


def remove_files(*args) -> None:
    final_args = [x for x in args if x is not None]
    for input_directory in final_args:
        if os.path.isfile(input_directory):
            os.remove(input_directory)


def find_latest_file(input_directory, ends_with="",is_dir=False):
    all_files = os.listdir(input_directory)
    if ends_with:
        all_files = [x for x in all_files if x.endswith(ends_with)]
    elif is_dir:
        all_files = [x for x in all_files if os.path.isdir(os.path.join(input_directory,x))]

    return sorted(all_files)[-1]
