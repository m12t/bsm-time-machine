"""
a util file for building test resource directories.
"""
import argparse
import os


def main(name: str):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../data")
    os.mkdir(name)
    os.chdir(name)
    os.mkdir("cases")
    os.mkdir("expected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dir_maker",
        description="setup the boiletplate dir structure for a test file. example command: `python dir_maker.py --name 'some_new_test'`",
    )
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    main(args.name)
