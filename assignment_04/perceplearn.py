import sys


def main(input_file_path: str):
    print(input_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python perceplearn.py <input_file>")
        exit(1)
    input_file_path = sys.argv[1]
    main(input_file_path)
